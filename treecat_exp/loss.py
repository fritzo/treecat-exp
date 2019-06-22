from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from treecat_exp.util import to_list

from pyro.contrib.tabular.features import Boolean, Discrete, Real


def reconstruction_loss_function(reconstructed, original, features, reduction="mean"):
    """
    reconstruction loss function as defined in Yoon et al, 2018.
    """
    loss = 0
    data_index = 0

    for f in features:
        if isinstance(f, Discrete):
            # discrete variable
            target = original[:, data_index: data_index + f.cardinality]
            loss += F.cross_entropy(reconstructed[:, data_index:data_index +
                                                  f.cardinality],
                                    target.max(-1).indices,  # XXX missing values will be 0?
                                    reduction=reduction)
            data_index += f.cardinality
        elif isinstance(f, Boolean):
            loss += F.binary_cross_entropy(reconstructed[:, data_index],
                                           original[:, data_index],
                                           reduction=reduction)
            data_index += 1
        else:
            assert isinstance(f, Real)
            # continuous variable
            loss += F.mse_loss(reconstructed[:, data_index], original[:, data_index],
                               reduction=reduction)
            data_index += 1
    return loss
