from __future__ import absolute_import, division, print_function

from pyro.contrib.tabular.features import Real

import torch


class Whitener(object):
    def __init__(self, features, data, mask):
        self.stats = []
        for f, d, m in zip(features, data, mask):
            if type(f) == Real:
                slice_f = d[m]
                mean_f, std_f = slice_f.mean().item(), slice_f.std().item()
                self.stats.append((mean_f, std_f))
            else:
                self.stats.append(None)

    def whiten(self, data, mask):
        whitened_data = []

        for d, m, s in zip(data, mask, self.stats):
            if s is not None:
                mean, std = s
                d[m] -= mean
                d[m] /= std
            whitened_data.append(d)

        return whitened_data

    def unwhiten(self, data, mask):
        unwhitened_data = []

        for d, m, s in zip(data, mask, self.stats):
            if s is not None:
                mean, std = s
                d[m] *= std
                d[m] += mean
            unwhitened_data.append(d)

        return unwhitened_data
