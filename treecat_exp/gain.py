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
from treecat_exp.layers import MixedActivation

from pdb import set_trace as bb


"""
Yoon, et al. GAIN: Missing Data Imputation using Generative Adversarial Nets. 2018.
"""


class Generator(nn.Module):
    def __init__(self, out_size, hidden_sizes, features):
        super(Generator, self).__init__()
        self.features = features
        previous_layer_size = out_size * 2
        hidden_layers = []
        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(nn.Tanh())
            previous_layer_size = layer_size
        hidden_layers.append(nn.Linear(previous_layer_size, out_size))
        self.out_layer = MixedActivation()
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, inputs, mask, training=False):
        inputs = torch.cat((inputs, mask), dim=1)
        hidden = self.hidden_layers(inputs)
        return self.out_layer(hidden, self.features, training=training, gumbel=False)

    def impute(self, inputs, mask, training=False):
        """
        Like ``.forward()``, but ensures that observed values are untouched.
        """
        out = self(inputs, mask, training=training)
        # fill in missing values with predicted reconstructed values
        out = out + mask * (inputs - out)
        return out


class Discriminator(nn.Module):
    def __init__(self, out_size, hidden_sizes):
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
    def __init__(self, generator, discriminator, features, whitener):
        self.generator = generator
        self.discriminator = discriminator
        self.one_hot = OneHotEncoder(features)
        self.features = features
        self.whitener = whitener

    def impute(self, data, mask, iterative=False):
        data = self.whitener.whiten(data, mask)
        data, mask = self.one_hot.encode(data, mask)
        data, t_mask = to_dense(data, mask)
        reconstruction = self.generator.impute(data, t_mask)
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
    generator = Generator(num_features, args.gen_layer_sizes, features)
    discriminator = Discriminator(num_features, args.disc_layer_sizes)
    if args.cuda and torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
    optim_g = SGD(generator.parameters(), lr=args.learning_rate)
    optim_d = SGD(discriminator.parameters(), lr=args.learning_rate)
    losses_g = []
    losses_d = []

    logging.info('TRAINING DISCRIMINATOR')
    for i in range(args.num_epochs):
        # first optimize the disc wrt a fixed gen...
        epoch_loss = 0
        num_batches = 0
        stop = True
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
                gen_data = generator.impute(batch_data, batch_mask, training=True)
            if torch.isnan(gen_data).any().item():
                logging.debug("NaN in generated data")

            pred = discriminator(gen_data.detach(), hint)
            loss = F.binary_cross_entropy(pred, batch_mask)
            loss.backward()
            optim_d.step()
            losses_d.append(loss)
            epoch_loss += loss
            num_batches += 1
        if i % args.logging_interval == 0:
            logging.info('[discriminator] epoch {}: loss = {}'
                         .format(i, epoch_loss))

    logging.info('TRAINING GENERATOR')
    for i in range(args.num_epochs):
        # ...then train the generator against the trained discriminator
        epoch_loss = 0
        num_batches = 0
        stop = True
        for batch_data_list, batch_mask_list in partition_data(data, mask, args.batch_size):
            optim_g.zero_grad()
            # preprocessing the data (TODO move this to preprocess.py)
            batch_data, batch_mask = to_dense(batch_data_list, batch_mask_list)
            hint = generate_hint(batch_mask, features, args.hint, args.hint_method)

            if args.cuda and torch.cuda.is_available():
                batch_data = to_cuda(batch_data)
                batch_mask = to_cuda(batch_mask)
                hint = to_cuda(hint)

            inverted_mask = 1 - batch_mask
            gen_data = generator(batch_data, batch_mask, training=True)
            imputed_data = batch_data * batch_mask + inverted_mask * gen_data
            pred = discriminator(imputed_data, hint)

            if torch.isnan(gen_data).any().item():
                logging.debug("NaN in generated data")
                bb()

            # we want to fool the discriminator now
            loss = F.binary_cross_entropy(pred, inverted_mask)
            # TODO downweight the reconstruction loss term as hyperparam?
            loss += reconstruction_loss_function(batch_mask * gen_data,
                                                 batch_mask * batch_data,
                                                 features,
                                                 reduction="sum") / torch.sum(batch_mask)
            if args.verbose and i % 10 == 0 and stop:
                missing_data = torch.stack([x.float() for x in batch_data_list], -1)
                if args.cuda:
                    missing_data = to_cuda(missing_data)
                missing_loss = reconstruction_loss_function(inverted_mask * gen_data,
                                                            inverted_mask * missing_data,
                                                            features,
                                                            reduction="sum") / torch.sum(inverted_mask)
                logging.debug("imputation loss = {}".format(missing_loss.item()))
                stop = False
            loss.backward()
            optim_g.step()
            losses_g.append(loss)
            epoch_loss += loss
            num_batches += 1
        if i % args.logging_interval == 0:
            logging.info('[generator] epoch {}: loss = {}'
                         .format(i, epoch_loss / num_batches))
    model = GAINModel(generator, discriminator, features, whitener)
    logging.info("saving object to: {}/{}.model.pkl".format(TRAIN, name))
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


def load_gain(name):
    return load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
