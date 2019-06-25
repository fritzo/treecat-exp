from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from treecat_exp.util import to_list

from pyro.contrib.tabular.features import Boolean, Discrete, Real


class MixedActivation(nn.Module):
    """
    output activation layer that usese:
    1) gumbel softmax for discrete
    2) sigmoid softmax for boolean
    3) tanh for real
    """
    def __init__(self):
        super(MixedActivation, self).__init__()

    def forward(self, inputs, features, training=False):
        """
        if training == true, output softmax.
        else input the argmax
        """
        data_index = 0
        output = []

        for f in features:
            if isinstance(f, Discrete):
                # discrete variable
                out = F.gumbel_softmax(inputs[:, data_index:data_index + f.cardinality],
                                       hard=not training,
                                       dim=-1)
                output.append(out)
                data_index += f.cardinality
            elif isinstance(f, Boolean):
                # discrete variable with cardinality = 1
                out = torch.sigmoid(inputs[:, data_index])
                assert len(out) == len(inputs)
                output.append(out.unsqueeze(1))
                data_index += 1
            else:
                # continuous variable
                out = torch.tanh(inputs[:, data_index])
                output.append(out.unsqueeze(1))
                data_index += 1
        return torch.cat(output, dim=1)
