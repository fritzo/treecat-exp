from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from sklearn import linear_model

from pyro.contrib.tabular import Discrete


class Regressor(object):
    """
    Object to train a regressor on imputed data and then test it on
    fully-observed data.
    """
    def __init__(self, features, out_feature, imputer, quantile):
        assert isinstance(out_feature, Discrete)
        assert 0 < quantile and quantile <= 1
        self.features = features
        self.out_feature = out_feature
        self.imputer = imputer
        self.quantile = quantile

        self.in_dims = []
        for i, feature in enumerate(features):
            if feature is out_feature:
                self.out_pos = i
            else:
                if isinstance(feature, Discrete):
                    self.in_dims.append(feature.cardinality)
                else:
                    raise NotImplementedError("TODO")
        self.in_dim = sum(self.in_dims)
        self.out_classes = np.array(range(out_feature.cardinality))
        self.predictor = linear_model.SGDClassifier()

    def _encode(self, data):
        for col in data:
            if col is not None:
                batch_size = len(col)
                break
        i = torch.arange(batch_size)
        X = torch.zeros(batch_size, self.in_dim)
        pos = 0
        for f, col, dim in zip(self.features, data, self.in_dims):
            if f is not self.out_feature:
                X[i, pos + col] = 1
                pos += dim
        return X

    def train(self, data):
        out_col = data[self.out_pos]
        while True:
            mask = torch.distributions.Bernoulli(self.quantile).sample((len(data),))
            mask[self.out_pos] = 0
            if mask.sum() >= 1 and (1 - mask).sum() >= 1:
                break
        masked_data = [col if m else None for (col, m) in zip(data, mask)]
        imputed_data = self.imputer(masked_data)
        imputed_data[self.out_pos] = None
        assert len(imputed_data) == len(data)
        X = self._encode(imputed_data).numpy()
        y = out_col.numpy()
        self.predictor.partial_fit(X, y, classes=self.out_classes)

    def test(self, data):
        out_col = data[self.out_pos]
        X = self._encode(data).numpy()
        y = out_col.numpy()
        return self.predictor.score(X, y)
