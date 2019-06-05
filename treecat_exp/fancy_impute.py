from __future__ import absolute_import, division, print_function

import os
import numpy as np

import pyro
import torch
from pyro.contrib.tabular.features import Real

from treecat_exp.util import CLEANUP, TRAIN, save_object, load_object, to_dense

from treecat_exp.whiten import Whitener
from fancyimpute import IterativeImputer


class FancyImputer(object):
    """
    n_iter: number of iterations to do IterativeImputer
    """
    def __init__(self, features, data, mask, method="IterativeImputer", n_iter=10):
        self.features = features
        self.whitener = Whitener(self.features, data, mask)
        self.method = method
        self.n_iter = n_iter
        assert method in ['IterativeImputer']

    def sample(self, data, mask):
        data = self.whitener.whiten(data, mask)

        data, tensor_mask = to_dense(data, mask)

        tensor_mask = tensor_mask.byte()
        data.masked_fill_(1 - tensor_mask, np.nan)

        data = data.data.numpy()
        ii = IterativeImputer(n_iter=self.n_iter,
                              min_value=-10.0, max_value=10.0)

        imputed_data = torch.from_numpy(ii.fit_transform(data)).t()
        imputed_data = [imputed_data[col] for col in range(imputed_data.size(0))]

        imputed_data = self.whitener.unwhiten(imputed_data, mask)

        return imputed_data


def train_fancy_imputer(name, features, data, mask, args):
    model = FancyImputer(features, data, mask,
                         method=args.fancy_method, n_iter=args.fancy_n_iter)
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


def load_fancy_imputer(name):
    return load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
