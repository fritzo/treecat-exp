from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
import torch.nn as nn
from torch.optim import Adam

from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TRAIN, interrupt, pdb_post_mortem
from util import to_cuda, reconstruction_loss_function
from multi import MultiOutput, SingleOutput, MultiInput

from pdb import set_trace as bb


class Decoder(nn.Module):
    def __init__(self, z_dim, output_size, hidden_sizes=[], variable_sizes=None, temperature=None):
        super(Decoder, self).__init__()
        hidden_activation = nn.Tanh()
        previous_layer_size = z_dim
        hidden_layers = []
        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if variable_sizes is None:
            self.output_layer = SingleOutput(previous_layer_size, output_size, activation=nn.Sigmoid())
        else:
            self.output_layer = MultiOutput(previous_layer_size, variable_sizes, temperature=temperature)

    def forward(self, code, training=False):
        if self.hidden_layers is None:
            return code
        return self.hidden_layers(code)


class Encoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_sizes=[], variable_sizes=None):
        super(Encoder, self).__init__()
        layers = []
        if variable_sizes is None:
            previous_layer_size = input_size
        else:
            multi_input_layer = MultiInput(variable_sizes)
            layers.append(multi_input_layer)
            previous_layer_size = multi_input_layer.size

        layer_sizes = list(hidden_sizes) + [z_dim]
        hidden_activation = nn.Tanh()

        for layer_size in layer_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.hidden_layers(inputs)


class VAE(nn.Module):
    def __init__(self, input_size, split_size, z_dim, encoder_hidden_sizes=[], decoder_hidden_sizes=[],
                 variable_sizes=None, temperature=None):

        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, split_size, hidden_sizes=encoder_hidden_sizes, variable_sizes=variable_sizes)
        self.decoder = Decoder(z_dim, input_size, hidden_sizes=decoder_hidden_sizes, variable_sizes=variable_sizes,
                               temperature=temperature)

        self.mu_layer = nn.Linear(split_size, z_dim)
        self.log_var_layer = nn.Linear(split_size, z_dim)

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, inputs, training=False):
        mu, log_var = self.encode(inputs)
        code = self.reparameterize(mu, log_var)
        reconstructed = self.decode(code, training=training)
        return code, reconstructed, mu, log_var

    def encode(self, inputs):
        outputs = self.encoder(inputs)
        return self.mu_layer(outputs), self.log_var_layer(outputs)

    def decode(self, code, training=False):
        return self.decoder(code, training=training)


def main(args):
    torch.manual_seed(1)
    # Load data.
    features, data, mask = load_data(args)
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("loaded {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))
    if args.multi:
        raise NotImplementedError('MultiInput/output')
    else:
        vae = VAE(args.batch_size, 128, 128, [30, 15], [30, 15])
    optim = Adam(vae.parameters(), lr=args.learning_rate)
    losses = []
    for i in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_data, batch_mask in partition_data(data, mask, args.batch_size):
            optim.zero_grad()
            if args.cuda and torch.cuda.is_available():
                batch_data = to_cuda(batch_data)
                batch_mask = to_cuda(batch_mask)
            bb()
            _, reconstructed, mu, log_var = vae(batch_data)
            # reconstruction loss + KLD
            reconstruction_loss = reconstruction_loss_function(mask * reconstructed,
                                                               mask * features,
                                                               None,  # variable sizes
                                                               reduction="sum") / torch.sum(mask)

            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kld_loss
            loss.backward()
            optim.step()
            losses.append(loss)
            epoch_loss += loss
            num_batches += 1


if __name__ == "__main__":
    """
    VAE imputation as implemented from Camino et al, 2018.
    run from toplevel treecat/ directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lending")
    parser.add_argument("--max-num-rows", default=9999999999, type=int)
    parser.add_argument("--iterative", action="store_true", default=False, help="iterative imputation")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--multi", action="store_true", default=False,
                        help="whether to use multi input/output per Camino et al (2018)")
    parser.add_argument("-i", "--init-size", default=1000000000, type=int)
    args = parser.parse_args()
    main(args)
