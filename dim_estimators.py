import pickle

import numpy as np
from scipy.spatial import distance_matrix
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from dimensions import (
    intrinsic_dim_sample_wise_double_mle,
)
from likelihood_estimators import LLGaussianMixtures, LLFlow, split_dataset
from src.cnf_lib import layers
from src.cnf_lib.layers.wrappers.cnf_regularization import l1_regularzation_fn
from src.cnf_lib.priors import GaussianMixture, Normal


def mle_skl(data, k):
    print("Computing the KNNs")
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(data)
    dist = nn.kneighbors(data)[0]
    mle, invmle = intrinsic_dim_sample_wise_double_mle(k, dist)
    return mle


def mle_inv(data, k):
    print("Computing the KNNs")
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(data)
    dist = nn.kneighbors(data)[0]
    mle, invmle = intrinsic_dim_sample_wise_double_mle(k, dist)
    return [1.0 / invmle.mean()] * data.shape[0]


def corr_dim(data, l_perc=0.000001, u_perc=0.01):
    N = len(data)
    distances = distance_matrix(data, data, p=2)[np.triu_indices(N, k=1)]
    r_low, r_high = np.quantile(distances, [l_perc, u_perc])
    C_r_list = []
    r_list = np.linspace(r_low, r_high, 3)

    for r in tqdm(r_list):
        distances_r = distances <= r
        # print(f'total, r = {r}, percenttrue: {(distances_r.sum())/distances_r.size}')
        C_r = 2 * distances_r.sum() / N / (N - 1)
        C_r_list.append(C_r)

    regr = linear_model.LinearRegression()
    regr.fit(np.log10(r_list).reshape(-1, 1), np.log10(C_r_list))
    return [regr.coef_[0]] * N


class LIDL:
    def __init__(self, model_type, **model_args):
        if model_type == "gm":
            self.model = LLGaussianMixtures(**model_args)
        elif model_type == "rqnsf":
            self.model = LLFlow(flow_type="rqnsf", **model_args)
        elif model_type == "maf":
            self.model = LLFlow(flow_type="maf", **model_args)
        else:
            raise ValueError(f"incorrect model type: {model_type}")

    def __call__(self, deltas, train_dataset, test):
        total_dim = train_dataset.shape[1]
        sort_deltas = np.argsort(np.array(deltas))
        lls = list()
        losses = list()
        tq = tqdm(deltas, position=0, leave=False, unit="delta")
        for delta in tq:
            tq.set_description(f"delta: {delta}")
            ll, score = self.model(delta=delta, dataset=train_dataset, test=test)
            lls.append(ll)
            losses.append(score)
        lls = np.array(lls)
        print(f"loss: {sum(losses)/len(losses)}")

        lls = lls[sort_deltas]
        deltas = np.array(deltas)[sort_deltas]

        lls = lls.transpose()
        dims = list()
        for i in range(lls.shape[0]):
            good_inds = ~np.logical_or(np.isnan(lls[i]), np.isinf(lls[i]))
            if ~good_inds.all():
                print(f"[WARNING] some log likelihoods are incorrect, deltas: {deltas}")
            ds = np.log(deltas[good_inds])
            ll = lls[i][good_inds]
            if ll.size < 2:
                dims.append(np.nan)
            else:
                regr = linear_model.LinearRegression()
                regr.fit(ds.reshape(-1, 1), ll)
                regr.predict(ds.reshape(-1, 1))
                dims.append(total_dim - regr.coef_[0])

        return np.array(dims)

def build_cnf(dims,regularization_fns):
    diffeq = layers.ODEnet(
        hidden_dims=(32,32,32),
        input_shape=(dims,),
        strides=None,
        conv=False,
    )
    odefunc = layers.ODEfunc(
        diffeq=diffeq,
        divergence_fn="brute_force",
        residual=False,
        rademacher=True,
    )
    cnf = layers.CNF(
        odefunc=odefunc,
        T=1.0,
        train_T=True,
        regularization_fns=regularization_fns,
        solver='dopri5',
    )
    return cnf

def build_sequential(dims, regularization_fns, num_blocks=2, batch_norm=False):
    chain = [build_cnf(dims, regularization_fns) for _ in range(num_blocks)]
    # if batch_norm:
    #     bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(num_blocks)]
    #     bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
    #     for a, b in zip(chain, bn_layers):
    #         bn_chain.append(a)
    #         bn_chain.append(b)
    #     chain = bn_chain
    return layers.SequentialFlow(chain)


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))
    for module in model.modules():
        if isinstance(module, layers.CNF):
            acc_reg_states = tuple(acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states()))
    return acc_reg_states

class CNF:
    def __init__(self, device, dims, lr, num_layers, run, epochs, bs):
        self.device = device
        self.val_size = 0.05
        # self.model = build_cnf(dims=dims).to(self.device)
        self.regularization_coeffs = [0.0001]
        self.regularization_funs = [l1_regularzation_fn]
        self.model = build_sequential(dims=dims, num_blocks=num_layers, regularization_fns=self.regularization_funs).to(self.device)
        self.epochs = epochs
        self.batch_size = bs
        self.lr = lr
        self.eps = 1.0
        self.run = run

        self.prior = GaussianMixture(dims=dims)
        # self.prior = Normal()
        print(f"Initial prior:\n{self.prior}")

    def get_loss(self, x):
        zero = torch.zeros(x.shape[0], 1).to(x)
        # transform to z
        z, delta_logp = self.model(x, zero)

        # compute log q(z)
        loss = delta_logp - self.prior.log_density(z, eps=self.eps).sum(1, keepdim=True)
        if len(self.regularization_coeffs) > 0:
            reg_states = get_regularization(self.model, self.regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, self.regularization_coeffs) if coeff != 0
            )
            loss = loss + reg_loss

        loss = torch.mean(loss)
        return loss

    def log_prob(self, x):
        z = self.model(x, None)
        logpz = self.prior.log_density(z, eps=self.eps).sum(1, keepdim=True)
        x, logprob = self.model(z, logpz, reverse=True)
        return logprob
    def __call__(self, train_dataset, test):
        train, val = split_dataset(train_dataset, self.val_size)
        if test.shape[1] != train_dataset.shape[1]:
            raise ValueError(
                f"train and test datasets have different number of features: \
                    train features: {train_dataset.shape[1]}, test features: {test.shape[1]}"
            )

        flow = self.model

        verbose = True
        optimizer = optim.Adam(list(flow.parameters()) + list(self.prior.parameters()), lr=self.lr)

        train_tensor = torch.tensor(train, dtype=torch.float32)
        val_tensor = torch.tensor(val, dtype=torch.float32, device=self.device)
        test_tensor = torch.tensor(test, dtype=torch.float32, device=self.device)

        best_loss = np.inf
        best_epoch = 0

        losses = list()
        results = list()
        if verbose:
            tq1 = tqdm.tqdm(range(self.epochs), position=1, leave=False)
        else:
            tq1 = range(self.epochs)

        for epoch in tq1:
            if verbose:
                tq1.set_description(f"epoch: {epoch + 1}")
            tq2 = DataLoader(train_tensor, batch_size=self.batch_size)

            # eps_bot = max((1.0 - epoch / 50), 0.01)**0.5
            eps_bot = max((1.0 - epoch / 50) * 0.1, 0.01)**0.5
            eps_top = 1.0  #  min(eps_bot + 0.1,  1.0)
            self.eps = 0.01**0.5 #eps_bot # np.random.uniform(eps_bot, eps_top)

            for x in tq2:
                x = x.to(self.device)
                optimizer.zero_grad()
                loss = self.get_loss(x)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                # validation loss for early stopping
                # val_loss = -flow.log_prob(inputs=val_tensor).mean()
                val_loss = self.get_loss(val_tensor).detach().cpu().numpy()
                losses.append(val_loss)

                # remember the results for early stopping
                # ll = -flow.log_prob(test_tensor)
                # results.append(ll.detach().cpu().numpy())

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch

                # if (epoch - best_epoch) > round(self.epochs * 2 / 100):
                #     print(f"Stopping after {best_epoch} epochs")
                #     return results[best_epoch], losses[best_epoch]
            if verbose:
                tq1.set_postfix_str(f"best loss: {losses[best_epoch]:.5f} | loss: {val_loss:.5f}")

            self.run["train/loss"].append(loss.detach().cpu().numpy())
            self.run["val/loss"].append(val_loss)
            self.run["num_evals"].append(self.model.chain[0].num_evals())

        logprob = self.log_prob(test_tensor)
        print(f"Loss: ", loss.detach().cpu().numpy())
        print(f"Val loss: ", val_loss)
        print(f"Final prior state:\n{self.prior}")

        z = self.model(test_tensor, None)
        densities = self.prior.log_density(z, eps=self.eps, raw=True)
        _, idx = densities.max(dim=1)

        return np.floor(idx.detach().cpu().numpy() / self.prior.multiplier), logprob.exp().detach().cpu().numpy()
