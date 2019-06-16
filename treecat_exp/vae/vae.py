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
from treecat_exp.whiten import Whitener
from treecat_exp.onehot import OneHotEncoder
from treecat_exp.vae.multi import MultiOutput, SingleOutput, MultiInput

from pdb import set_trace as bb


class Decoder(nn.Module):
    def __init__(self, z_dim, output_size, hidden_sizes=[]):
        super(Decoder, self).__init__()
        previous_layer_size = z_dim
        hidden_layers = []
        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(nn.ReLU())
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        self.output_layer = SingleOutput(previous_layer_size, output_size, activation=nn.Sigmoid())

    def forward(self, code, training=False):
        if self.hidden_layers is None:
            return code
        return self.hidden_layers(code)


class Encoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_sizes=[]):
        super(Encoder, self).__init__()
        layers = []
        previous_layer_size = input_size

        layer_sizes = list(hidden_sizes) + [z_dim]

        for layer_size in layer_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(nn.ReLU())
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
    def __init__(self, input_size, z_dim, encoder_hidden_sizes=[], decoder_hidden_sizes=[]):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, z_dim, hidden_sizes=encoder_hidden_sizes)
        self.decoder = Decoder(z_dim, input_size, hidden_sizes=decoder_hidden_sizes)

    def forward(self, inputs, training=False):
        z, mu, log_var = self.encoder(inputs)
        reconstructed = self.decoder(z, training=training)
        return z, reconstructed, mu, log_var


def impute(name, data, mask, args):
    vae_model = load_vae(name)

    data, mask = to_dense(data, mask)
    if args.cuda and torch.cuda.is_available():
        data = to_cuda(data)
        mask = to_cuda(mask)
    inverted_mask = 1 - mask
    observed = data * mask
    missing = torch.randn_like(data)

    if args.noise_lr is not None:
        missing = torch.randn_like(data, requires_grad=True)
        optim = Adam([missing], weight_decay=0, lr=args.noise_lr)
    for i in range(args.num_epochs):
        if args.noise_lr is not None:
            optim.zero_grad()

        noisy_data = observed + missing * inverted_mask
        _, reconstructed, _, _ = vae_model.vae(noisy_data, training=True)

        if torch.isnan(reconstructed).any():
            raise ValueError('reconstructed tensor has NaNs')

        observed_loss = reconstruction_loss_function(mask * reconstructed.clamp(min=0., max=1.),
                                                     mask * data.clamp(min=0., max=1.),
                                                     None,  # multioutput
                                                     reduction="sum") / torch.sum(mask)
#         missing_loss = reconstruction_loss_function(inverted_mask * reconstructed.clamp(min=0., max=1.),
#                                                     inverted_mask * data.clamp(min=0., max=1.),
#                                                     None,  # multioutput
#                                                     reduction="sum") / torch.sum(mask)

        if args.noise_lr is None:
            missing = reconstructed * inverted_mask
        else:
            observed_loss.backward()
            optim.step()

        if i % args.logging_interval == 0:
            logging.info('epoch {} loss = {}'.format(i, observed_loss.item()))

        if observed_loss < args.tolerance:
            break

        save_object(vae_model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
        return vae_model


class VAEModel(object):
    def __init__(self, vae, features, whitener):
        self.vae = vae
        self.one_hot = OneHotEncoder(features)
        self.whitener = whitener

    def sample(self, data, mask):
        data = self.whitener.whiten(data, mask)
        data, mask = self.one_hot.encode(data, mask)
        data, _ = to_dense(data, mask)
        reconstruction = self.vae(data)[1]
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
    if args.multi:
        raise NotImplementedError('MultiInput/output')
    else:
        # len(data) > len(features) if discretes were one-hot encoded
        vae = VAE(len(data), args.hidden_dim, args.encoder_layer_sizes, args.decoder_layer_sizes + [len(data)])
    if args.cuda and torch.cuda.is_available():
        vae.cuda()
    optim = Adam(vae.parameters(), lr=args.learning_rate)
    losses = []
    for i in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_data, batch_mask in partition_data(data, mask, args.batch_size):
            optim.zero_grad()
            # preprocessing the data (TODO move this to preprocess.py)
            batch_data, batch_mask = to_dense(batch_data, batch_mask)
            if args.cuda and torch.cuda.is_available():
                batch_data = to_cuda(batch_data)
                batch_mask = to_cuda(batch_mask)
            _, reconstructed, mu, log_var = vae(batch_data)
            reconstruction_loss = reconstruction_loss_function(batch_mask * reconstructed,
                                                               batch_mask * batch_data,
                                                               features,  # multioutput
                                                               reduction="sum") / torch.sum(batch_mask)

            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kld
            loss.backward()
            optim.step()
            losses.append(loss)
            epoch_loss += loss
            num_batches += 1
#             if num_batches % args.logging_interval == 0:
#                 data_size = data[0].shape[0]
#                 logging.info('[batch {}/{}]: loss = {}'
#                              .format(min(num_batches * args.batch_size, data_size),
#                                      data_size,
#                                      epoch_loss))
        if i % args.logging_interval == 0:
            logging.info('epoch {}: loss = {}'
                         .format(i, epoch_loss))
    model = VAEModel(vae, features, whitener)
    logging.info("saving object to: {}/{}.model.pkl".format(TRAIN, name))
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


def load_vae(name):
    return load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
