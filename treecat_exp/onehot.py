from __future__ import absolute_import, division, print_function

import os
import warnings

import numpy as np
import pyro
import torch
from pyro.contrib.tabular.features import Boolean, Discrete, Real


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
