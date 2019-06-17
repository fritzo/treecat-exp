from __future__ import absolute_import, division, print_function

import os
import argparse
import logging

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn import functional as F

from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TRAIN, interrupt, pdb_post_mortem, save_object, load_object, to_dense, to_list, to_cuda
from treecat_exp.loss import reconstruction_loss_function, generate_hint
from treecat_exp.whiten import Whitener
from treecat_exp.onehot import OneHotEncoder
from treecat_exp.vae.multi import SingleOutput

from pdb import set_trace as bb


"""
Yoon, et al. GAIN: Missing Data Imputation using Generative Adversarial Nets. 2018.
"""


class Generator(nn.Module):
    def __init__(self, size, hidden_sizes=[], mask_variables=False, temperature=None):
        super(Generator, self).__init__()
        self.multi_input_layer = None
        previous_layer_size = size * 2
        hidden_layers = []
        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(nn.Tanh())
            previous_layer_size = layer_size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = SingleOutput(previous_layer_size, size, activation=nn.Sigmoid())

    def forward(self, inputs, mask, training=False):
        inputs = torch.cat((inputs, mask), dim=1)
        outputs = self.hidden_layers(inputs)
        return self.output_layer(outputs, training=training)


class Discriminator(nn.Module):
    def __init__(self, out_size, hidden_sizes=[], hint_variables=False):
        super(Discriminator, self).__init__()
        previous_layer_size = out_size * 2
        layers = []
        for layer_size in hidden_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(nn.Tanh())
            previous_layer_size = layer_size
        layers.append(nn.Linear(previous_layer_size, out_size))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs, hints):
        inputs = torch.cat((inputs, hints), dim=1)
        return self.layers(inputs)


class GAINModel(object):
    def __init__(self, gan, features, whitener):
        self.gan = gan
        self.one_hot = OneHotEncoder(features)
        self.features = features
        self.whitener = whitener

    def sample(self, data, mask, iterative=False):
        data = self.whitener.whiten(data, mask)
        data, mask = self.one_hot.encode(data, mask)
        data, t_mask = to_dense(data, mask)
        reconstruction = self.vae(data)[1]
        reconstruction, mask = self.one_hot.decode(to_list(reconstruction), mask)
        unwhitened = self.whitener.unwhiten(reconstruction, mask)
        return unwhitened


def train_gain(name, features, data, mask, args):
    logging.basicConfig(format="%(relativeCreated) 9d %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    torch.manual_seed(args.seed)
    # normalize data
    whitener = Whitener(features, data, mask)
    one_hot = OneHotEncoder(features)
    data = whitener.whiten(data, mask)
    data, mask = one_hot.encode(data, mask)

    # len(data) > len(features) if discretes were one-hot encoded
    num_features = len(data)
    generator = Generator(num_features, args.gen_layer_sizes)
    discriminator = Discriminator(num_features, args.disc_layer_sizes)
    if args.cuda and torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
    optim_g = SGD(generator.parameters(), lr=args.learning_rate)
    optim_d = SGD(discriminator.parameters(), lr=args.learning_rate)
    losses = []

    for i in range(args.num_epochs):
        # first optimize the disc with a fixed gen
        epoch_loss = 0
        num_batches = 0
        for batch_data, batch_mask in partition_data(data, mask, args.batch_size):
            optim_d.zero_grad()
            # preprocessing the data (TODO move this to preprocess.py)
            batch_data, batch_mask = to_dense(batch_data, batch_mask)
            hint = generate_hint(batch_mask, features, args.hint, args.hint_method)
            if args.cuda and torch.cuda.is_available():
                batch_data = to_cuda(batch_data)
                batch_mask = to_cuda(batch_mask)
                hint = to_cuda(hint)
            with torch.no_grad():
                gen_data = generator(batch_data, batch_mask)
            pred = discriminator(gen_data, hint)
            loss = F.binary_cross_entropy(pred, batch_mask)
            loss.backward()
            optim_d.step()
            losses.append(loss)
            epoch_loss += loss
            num_batches += 1
        if i % args.logging_interval == 0:
            logging.info('epoch {}: loss = {}'
                         .format(i, epoch_loss))
    # then train generator against trained discriminator
    for i in range(args.num_epochs):
        pass
    model = GAINModel(gain, features, whitener)
    logging.info("saving object to: {}/{}.model.pkl".format(TRAIN, name))
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


def load_gain(name):
    return load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
