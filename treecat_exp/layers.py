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

    def forward(self, inputs, features, training=False, temp=None):
        """
        if training == true, output softmax.
        else output the argmax
        """
        data_index = 0
        output = []

        for f in features:
            if isinstance(f, Discrete):
                # discrete variable
                if temp:
                    out = F.gumbel_softmax(inputs[:, data_index:data_index + f.cardinality],
                                           tau=temp,
                                           hard=not training,
                                           dim=-1)
                else:
                    out = F.softmax(inputs[:, data_index:data_index + f.cardinality],
                                    dim=-1)
                    if not training:
                        x = torch.zeros_like(out)
                        x[torch.arange(len(x)), out.argmax(-1).long()] = 1.
                        out = x
                output.append(out)
                data_index += f.cardinality
            elif isinstance(f, Boolean):
                # discrete variable with cardinality = 1
                out = torch.sigmoid(inputs[:, data_index])
                assert len(out) == len(inputs)
                output.append(out.unsqueeze(1))
                data_index += 1
            else:
                assert isinstance(f, Real)
                # continuous variable
                out = torch.tanh(inputs[:, data_index])
                output.append(out.unsqueeze(1))
                data_index += 1
        return torch.cat(output, dim=1)
