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


def generate_hint(mask, features, prob, method='drop'):
    # there is a difference between the code implementation and what is described in the paper
    # in the code, the hint drops 1s in the mask with probability `prob` (method='drop')
    # https://github.com/jsyoon0823/GAIN/blob/master/MNST_Code_Example.py#L44
    # in the paper, zero values in the mask are replaced with 0.5 given probability `probs`
    # http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf
    assert isinstance(mask, torch.Tensor)
    hints = []
    if method == 'drop':
        # method in the code
        for f in features:
            hint = (torch.rand((len(mask), 1), device=mask[0].device) < prob).float()
            if isinstance(f, Discrete):
                hint = hint.repeat(1, f.cardinality)
            hints.append(hint)
        return torch.cat(hints, dim=1) * mask
    else:
        # method in the paper
        replace = torch.distributions.Categorical(torch.ones(mask.shape)).sample().tolist()
        idx = torch.arange(len(mask)).tolist()
        hint = mask.clone()
        hint[idx, replace] = 0.5
        return hint
