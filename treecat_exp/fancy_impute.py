from __future__ import absolute_import, division, print_function

import os
import warnings

import numpy as np
import pyro
import torch
from fancyimpute import IterativeImputer, IterativeSVD, KNN
from pyro.contrib.tabular.features import Boolean, Discrete, Real

from treecat_exp.util import CLEANUP, TRAIN, load_object, save_object, to_dense
from treecat_exp.whiten import Whitener
from treecat_exp.onehot import OneHotEncoder


class FancyImputer(object):
    """
    n_iter: number of iterations to do IterativeImputer
    """
    def __init__(self, features, data, mask, method="IterativeImputer",
                 n_iter=10, svd_rank=10, knn_neighbors=5):
        self.features = features
        self.whitener = Whitener(features, data, mask)
        self.encoder = OneHotEncoder(features)
        self.method = method
        self.n_iter = n_iter
        self.svd_rank = svd_rank
        self.knn_neighbors = knn_neighbors
        assert method in ['IterativeImputer', 'IterativeSVD', 'KNN']

    def impute(self, data, mask):
        data = self.whitener.whiten(data, mask)
        data, mask = self.encoder.encode(data, mask)

        # Note this transposes the data.
        data, tensor_mask = to_dense(data, mask)

        tensor_mask = tensor_mask.byte()
        data.masked_fill_(1 - tensor_mask, np.nan)

        data = data.data.numpy()
        if self.method == 'IterativeImputer':
            ii = IterativeImputer(n_iter=self.n_iter,
                                  min_value=-10.0, max_value=10.0)
        elif self.method == 'IterativeSVD':
            ii = IterativeSVD(max_iters=self.n_iter, rank=self.svd_rank,
                              min_value=-10.0, max_value=10.0)
        elif self.method == 'KNN':
            ii = KNN(k=self.knn_neighbors,
                     min_value=-10.0, max_value=10.0)
        else:
            raise ValueError("Unknown method: {}".format(self.method))

        # Note that IterativeImputer weirdly drops all-nan columns.
        ok = (data == data).any(0)
        if not ok.all():
            warnings.warn("Encountered all-nan column in dataset")
        imputed_data = np.zeros(data.shape)
        imputed_data[:, ok] = ii.fit_transform(data[:, ok])
        imputed_data = list(torch.from_numpy(imputed_data).t())

        imputed_data, mask = self.encoder.decode(imputed_data, mask)
        imputed_data = self.whitener.unwhiten(imputed_data, mask)
        return imputed_data


def train_fancy_imputer(name, features, data, mask, args):
    model = FancyImputer(features, data, mask,
                         method=args.fancy_method, n_iter=args.fancy_n_iter,
                         svd_rank=args.fancy_svd_rank, knn_neighbors=args.fancy_knn_neighbors)
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


def load_fancy_imputer(name):
    return load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
