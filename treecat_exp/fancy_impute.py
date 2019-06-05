from __future__ import absolute_import, division, print_function

import os
import warnings

import numpy as np
import pyro
import torch
from fancyimpute import IterativeImputer
from pyro.contrib.tabular.features import Boolean, Discrete, Real

from treecat_exp.util import CLEANUP, TRAIN, load_object, save_object, to_dense
from treecat_exp.whiten import Whitener


class OneHotEncoder(object):
    def __init__(self, features):
        self.features = features

    def encode(self, data, mask):
        assert len(data) == len(self.features)
        prototype = next(c for c in data if c is not None)
        result_data = []
        result_mask = []
        for f, d, m in zip(self.features, data, mask):
            if d is None:
                d = prototype.new_zeros(len(prototype))
            if isinstance(f, Discrete):
                one_hot = d.new_zeros(f.cardinality, len(d))
                one_hot[d.long(), torch.arange(len(d))] = 1
                for d in one_hot:
                    result_data.append(d)
                    result_mask.append(m)
            else:
                result_data.append(d)
                result_mask.append(m)
        assert len(result_data) == len(result_mask)
        return result_data, result_mask

    def decode(self, data, mask):
        assert len(data) == len(mask)
        result_data = []
        result_mask = []
        pos = 0
        for f in self.features:
            if isinstance(f, Discrete):
                d = torch.stack(data[pos: pos + f.cardinality], -1)
                d = torch.max(d, -1)[1].float()  # argmax
                result_data.append(d)
                result_mask.append(mask[pos])
                pos += f.cardinality
            elif isinstance(f, Boolean):
                result_data.append(data[pos].round().clamp(min=0, max=1))
                result_mask.append(mask[pos])
                pos += 1
            else:
                result_data.append(data[pos])
                result_mask.append(mask[pos])
                pos += 1
        assert len(result_data) == len(self.features)
        assert len(result_mask) == len(self.features)
        return result_data, result_mask


class FancyImputer(object):
    """
    n_iter: number of iterations to do IterativeImputer
    """
    def __init__(self, features, data, mask, method="IterativeImputer", n_iter=10):
        self.features = features
        self.whitener = Whitener(features, data, mask)
        self.encoder = OneHotEncoder(features)
        self.method = method
        self.n_iter = n_iter
        assert method in ['IterativeImputer']

    def sample(self, data, mask):
        data = self.whitener.whiten(data, mask)
        data, mask = self.encoder.encode(data, mask)

        # Note this transposes the data.
        data, tensor_mask = to_dense(data, mask)

        tensor_mask = tensor_mask.byte()
        data.masked_fill_(1 - tensor_mask, np.nan)

        data = data.data.numpy()
        ii = IterativeImputer(n_iter=self.n_iter,
                              min_value=-10.0, max_value=10.0)

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
                         method=args.fancy_method, n_iter=args.fancy_n_iter)
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


def load_fancy_imputer(name):
    return load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
