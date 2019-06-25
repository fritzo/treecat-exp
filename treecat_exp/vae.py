from __future__ import absolute_import, division, print_function

import os
import argparse
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss

from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TRAIN, interrupt, pdb_post_mortem, save_object, load_object, to_dense, to_list, to_cuda
from treecat_exp.loss import reconstruction_loss_function
from treecat_exp.layers import MixedActivation
from treecat_exp.whiten import Whitener
from treecat_exp.onehot import OneHotEncoder

from pdb import set_trace as bb


class Decoder(nn.Module):
    def __init__(self, z_dim, output_size, features, hidden_sizes=[]):
        super(Decoder, self).__init__()
        self.features = features
        previous_layer_size = z_dim
        hidden_layers = []
        for i, layer_size in enumerate(hidden_sizes):
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            if i < len(hidden_sizes) - 1:
                hidden_layers.append(nn.Tanh())
            previous_layer_size = layer_size
        self.out_layer = MixedActivation()

        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, code, training=False):
        hidden = self.hidden_layers(code)
        return self.out_layer(hidden, self.features, training=training)


class Encoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_sizes=[]):
        super(Encoder, self).__init__()
        layers = []
        previous_layer_size = input_size

        layer_sizes = list(hidden_sizes) + [z_dim]

        for layer_size in layer_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(nn.Tanh())
            previous_layer_size = layer_size

        self.mu_layer = nn.Linear(z_dim, z_dim)
        self.log_var_layer = nn.Linear(z_dim, z_dim)

        self.hidden_layers = nn.Sequential(*layers)

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, inputs):
        h = self.hidden_layers(inputs)
        mu, log_var = self.mu_layer(h), self.log_var_layer(h)
        return self._reparameterize(mu, log_var), mu, log_var


class VAE(nn.Module):
    def __init__(self, input_size, z_dim, features, encoder_hidden_sizes=[], decoder_hidden_sizes=[]):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, z_dim, hidden_sizes=encoder_hidden_sizes)
        self.decoder = Decoder(z_dim, input_size, features, hidden_sizes=decoder_hidden_sizes)

    def forward(self, inputs, training=False):
        z, mu, log_var = self.encoder(inputs)
        reconstructed = self.decoder(z, training=training)
        return z, reconstructed, mu, log_var

    def impute(self, inputs, mask):
        """
        Like ``.forward()``, but ensures that observed values are untouched.
        """
        z, reconstructed, mu, log_var = self(inputs, training=False)
        # fill in missing values with predicted reconstructed values
        reconstructed += mask * (inputs - reconstructed)
        return reconstructed


class VAEModel(object):
    def __init__(self, vae, features, whitener):
        self.vae = vae
        self.one_hot = OneHotEncoder(features)
        self.whitener = whitener

    def sample(self, data, mask, iters=1):
        assert iters >= 1
        data = self.whitener.whiten(data, mask)
        data, mask = self.one_hot.encode(data, mask)
        data, t_mask = to_dense(data, mask)
        reconstruction = data
        for i in range(iters):
            reconstruction = self.vae.impute(reconstruction, t_mask)

        reconstruction, mask = self.one_hot.decode(to_list(reconstruction), mask)
        unwhitened = self.whitener.unwhiten(reconstruction, mask)
        return unwhitened

    def log_prob(self, data, mask):
        data = self.whitener.whiten(data, mask)
        data, mask = self.one_hot.encode(data, mask)
        data, _ = to_dense(data, mask)
        z, mu, log_var = self.vae.encoder(data)
        # TODO: correct?
        return torch.distributions.Normal(mu, (log_var / 2).exp()).log_prob(z).sum(-1)


def train_vae(name, features, data, mask, args):
    logging.basicConfig(format="%(relativeCreated) 9d %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    # normalize data
    whitener = Whitener(features, data, mask)
    one_hot = OneHotEncoder(features)
    data = whitener.whiten(data, mask)
    data, mask = one_hot.encode(data, mask)
    torch.manual_seed(args.seed)
    # len(data) > len(features) if discretes were one-hot encoded
    vae = VAE(len(data), args.hidden_dim, features, args.encoder_layer_sizes, args.decoder_layer_sizes + [len(data)])
    if args.cuda and torch.cuda.is_available():
        vae.cuda()
    optim = Adam(vae.parameters(), lr=args.learning_rate)
    losses = []
    for i in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0
        stop = True
        for batch_data_list, batch_mask_list in partition_data(data, mask, args.batch_size):
            optim.zero_grad()
            # preprocessing the data (TODO move this to preprocess.py)
            batch_data, batch_mask = to_dense(batch_data_list, batch_mask_list)
            if args.cuda and torch.cuda.is_available():
                batch_data = to_cuda(batch_data)
                batch_mask = to_cuda(batch_mask)

            _, reconstructed, mu, log_var = vae(batch_data, training=True)
            # reconstruction loss per data only on the observed values
            reconstruction_loss = reconstruction_loss_function(batch_mask * reconstructed,
                                                               batch_mask * batch_data,
                                                               features,
                                                               reduction="sum") / torch.sum(batch_mask)
            # fixed N(0,1) prior
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # scale kl by hyperparam
            loss = reconstruction_loss + args.kl_factor * kld
            if i % 10 == 0 and stop and args.verbose:
                inverted_mask = 1 - batch_mask
                missing_data = torch.stack([x.float() for x in batch_data_list], -1)
                if args.cuda:
                    missing_data = to_cuda(missing_data)
                imputation_loss = reconstruction_loss_function(inverted_mask * reconstructed,
                                                               inverted_mask * missing_data,
                                                               features,
                                                               reduction="sum") / torch.sum(inverted_mask)
                logging.debug("Epoch {}".format(i))
                logging.debug("recon_loss = {}".format(reconstruction_loss.item()))
                logging.debug("kld = {}".format((args.kl_factor * kld).item()))
                logging.debug("imputation loss = {}".format(imputation_loss.item()))
                stop = False
#                 bb()

            loss.backward()
            optim.step()
            losses.append(loss)
            epoch_loss += loss
            num_batches += 1
        if i % args.logging_interval == 0:
            logging.info('epoch {}: loss = {}'
                         .format(i, epoch_loss / num_batches))
    model = VAEModel(vae, features, whitener)
    logging.info("saving object to: {}/{}.model.pkl".format(TRAIN, name))
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


def load_vae(name):
    return load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
