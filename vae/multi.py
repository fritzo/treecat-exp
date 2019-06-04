from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import OneHotCategorical


class MultiInput(nn.Module):

    def __init__(self, variable_sizes, min_embedding_size=2, max_embedding_size=50):
        super(MultiInput, self).__init__()

        self.has_categorical = False
        self.size = 0

        embeddings = nn.ParameterList()
        for i, variable_size in enumerate(variable_sizes):
            if variable_size == 1:
                # real
                embeddings.append(None)
                self.size += 1
            else:
                # discrete (categorical)
                embedding_size = max(min_embedding_size, min(max_embedding_size, int(variable_size / 2)))
                embedding = nn.Parameter(data=torch.Tensor(variable_size, embedding_size).normal_(), requires_grad=True)

                embeddings.append(embedding)
                self.size += embedding_size
                self.has_categorical = True

        if self.has_categorical:
            self.variable_sizes = variable_sizes
            self.embeddings = embeddings

    def forward(self, inputs):
        if self.has_categorical:
            outputs = []
            start = 0
            for variable_size, embedding in zip(self.variable_sizes, self.embeddings):
                # extract the variable
                end = start + variable_size
                variable = inputs[:, start:end]
                if variable_size == 1:
                    # numerical variable (leave the input as it is)
                    outputs.append(variable)
                else:
                    # categorical variable
                    output = torch.matmul(variable, embedding).squeeze(1)
                    outputs.append(output)
                # move the variable limits
                start = end

            # concatenate all the variable outputs
            return torch.cat(outputs, dim=1)
        return inputs


class MultiOutput(nn.Module):

    def __init__(self, input_size, variable_sizes, temperature=None):
        super(MultiOutput, self).__init__()

        self.output_layers = nn.ModuleList()
        self.output_activations = nn.ModuleList()

        numerical_size = 0
        for i, variable_size in enumerate(variable_sizes):
            if variable_size > 1:
                # if it is a categorical variable
                if numerical_size > 0:
                    self.output_layers.append(nn.Linear(input_size, numerical_size))
                    self.output_activations.append(Activation(temperature=temperature))
                    numerical_size = 0
                self.output_layers.append(nn.Linear(input_size, variable_size))
                self.output_activations.append(Activation(temperature=temperature))
            # if not, accumulate numerical variables
            else:
                numerical_size += 1

        # create the remaining accumulated numerical layer
        if numerical_size > 0:
            self.output_layers.append(nn.Linear(input_size, numerical_size))
            self.output_activations.append(Activation(sigmoid=True))

    def forward(self, inputs, training=True, concat=True):
        outputs = []
        for output_layer, output_activation in zip(self.output_layers, self.output_activations):
            logits = output_layer(inputs)
            output = output_activation(logits, training=training)
            outputs.append(output)

        if concat:
            return torch.cat(outputs, dim=1)
        return outputs


class SingleOutput(nn.Module):

    def __init__(self, previous_layer_size, output_size, activation=None):
        super(SingleOutput, self).__init__()
        if activation is None:
            self.model = nn.Linear(previous_layer_size, output_size)
        else:
            self.model = nn.Sequential(nn.Linear(previous_layer_size, output_size), activation)

    def forward(self, hidden, training=False):
        return self.model(hidden)


class Activation(nn.Module):

    def __init__(self, temperature=None, sigmoid=False):
        super(Activation, self).__init__()
        self.temperature = temperature
        self.sigmopid = sigmoid

    def forward(self, logits, training=True):
        if self.sigmoid:
            return F.sigmoid(logits)
        if self.temperature is not None:
            return F.gumbel_softmax(logits, hard=not training, tau=self.temperature)
        elif training:
            return F.softmax(logits, dim=1)
        else:
            return OneHotCategorical(logits=logits).sample()
