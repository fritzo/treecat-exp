from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss

from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TRAIN, interrupt, pdb_post_mortem
from vae.util import to_cuda, reconstruction_loss_function
from vae.multi import MultiOutput, SingleOutput, MultiInput

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
    def __init__(self, input_size, z_dim, encoder_hidden_sizes=[], decoder_hidden_sizes=[],
                 variable_sizes=None, temperature=None):

        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, z_dim, hidden_sizes=encoder_hidden_sizes, variable_sizes=variable_sizes)
        self.decoder = Decoder(z_dim, input_size, hidden_sizes=decoder_hidden_sizes, variable_sizes=variable_sizes,
                               temperature=temperature)

    def forward(self, inputs, training=False):
        z, mu, log_var = self.encoder(inputs)
        reconstructed = self.decoder(z, training=training)
        return z, reconstructed, mu, log_var


def impute(args):
    vae, features, mask = None, None, None
    loss_function = MSELoss()
    inverted_mask = 1 - mask
    observed = features * mask
    missing = torch.randn_like(features)

    if args.noise_lr is not None:
        missing = torch.randn_like(features, requires_grad=True)
        optim = Adam([missing], weight_decay=0, lr=args.noise_lr)

    vae.train(mode=True)

    for iteration in range(args.num_epochs):
        if args.noise_lr is not None:
            optim.zero_grad()

        noisy_features = observed + missing * inverted_mask
        _, reconstructed, _, _ = vae(noisy_features, training=True)

        observed_loss = reconstruction_loss_function(mask * reconstructed.clamp(min=0., max=1.),
                                                     mask * features.clamp(min=0., max=1.),
                                                     None,  # multioutput
                                                     reduction="sum") / torch.sum(mask)
        missing_loss = reconstruction_loss_function(inverted_mask * reconstructed.clamp(min=0., max=1.),
                                                    inverted_mask * features.clamp(min=0., max=1.),
                                                    None,  # multioutput
                                                    reduction="sum") / torch.sum(mask)

        masked = mask * features + (1. - mask) * reconstructed
        loss = torch.sqrt(loss_function(masked, features))

        if args.noise_lr is None:
            missing = reconstructed * inverted_mask
        else:
            observed_loss.backward()
            optim.step()

        if observed_loss < args.tolerance:
            break

        return observed_loss, missing_loss, loss


class VAEModel(object):
    def __init__(self, vae):
        self.vae = vae

    def sample(self, data, mask):
        data = torch.stack(data, -1).float()
        mask = torch.stack(mask, -1).float()
        # TODO scale noise appropriately
        masked_data = data * mask + (1. - mask) * torch.randn(data.shape, device=mask.device)
        out = self.vae(masked_data)
        return out[1]  # reconstruction

    def log_prob(self, data, mask):
        data = torch.stack(data, -1).float()
        mask = torch.stack(mask, -1).float()
        masked_data = data * mask + (1. - mask) * torch.randn(data.shape, device=mask.device)
        z, mu, log_var = self.vae.encoder(masked_data)
        # TODO: correct?
        return torch.distributions.Normal(mu, (log_var / 2).exp()).log_prob(z).sum()


def train_vae(name, features, data, mask, args):
    torch.manual_seed(args.seed)
    if args.multi:
        raise NotImplementedError('MultiInput/output')
    else:
        vae = VAE(len(features), args.hidden_dim, [700, 128], [700, len(features)])
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        vae.cuda()
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
            # preprocessing the data (TODO move this to preprocess.py)
            for i, m in enumerate(batch_mask):
                if isinstance(m, torch.Tensor):
                    continue
                if not m:
                    # fill missing data with std noise
                    # TODO scale noise appropriately
                    batch_data[i] = torch.randn(args.batch_size)
                    batch_mask[i] = torch.zeros(args.batch_size)
                else:
                    batch_mask[i] = torch.ones(args.batch_size)
            batch_data = [torch.stack([x.float() for x in batch_data], -1)][0]
            batch_mask = [torch.stack([x.float() for x in batch_mask], -1)][0]
            _, reconstructed, mu, log_var = vae(batch_data)
            # reconstruction loss + KLD
            reconstruction_loss = reconstruction_loss_function(batch_mask * reconstructed.clamp(min=0., max=1.),
                                                               batch_mask * batch_data.clamp(min=0., max=1.),
                                                               None,  # multioutput
                                                               reduction="sum") / torch.sum(batch_mask)

            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kld
            loss.backward()
            optim.step()
            losses.append(loss)
            epoch_loss += loss
            num_batches += 1
            if i % args.logging_interval == 0:
                logging.info('[batch {}/{}]: loss = {}'
                             .format((num_batches * args.batch_size),
                                     data.shape[0],
                                     epoch_loss))
    return VAEModel(vae)


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
    parser.add_argument("-nlr", "--noise-lr", default=0.001, type=float, help="noise lr (for iterative imputation)")
    parser.add_argument("--tolerance", default=0.001, type=float, help="tolerance for iterative imputation")
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--impute", action="store_true", default=False)
    parser.add_argument("--multi", action="store_true", default=False,
                        help="whether to use multi input/output per Camino et al (2018)")
    parser.add_argument("-i", "--init-size", default=1000000000, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-l", "--logging-interval", default=10, type=int)
    args = parser.parse_args()
    logging.basicConfig(format="%(relativeCreated) 9d %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    if args.impute:
        impute(args)
