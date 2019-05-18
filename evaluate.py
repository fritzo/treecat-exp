from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys

import numpy as np
import torch
from six.moves import cPickle as pickle

import pyro
from pyro.infer import TraceEnum_ELBO
from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TEST, TRAIN, interrupt


class LossFunction(object):
    def __init__(self, model):
        self.model = model
        self.elbo = TraceEnum_ELBO(max_plate_nesting=1)

    @torch.no_grad()
    def __call__(self, data):
        return self.elbo.loss(self.model.model, self.model.guide, data)


def main(args):
    # Load data.
    features, data = load_data(args)
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("loaded {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))
    logging.info("\n".join(["Features:"] + [str(f) for f in features]))

    # Load a trained model.
    logging.debug("Loading model")
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)
    pyro.get_param_store().clear()
    pyro.get_param_store().load(os.path.join(TRAIN, "{}.model.pyro".format(args.dataset)))
    with open(os.path.join(TRAIN, "{}.model.pkl".format(args.dataset)), "rb") as f:
        model = pickle.load(f)

    def save(metrics):
        if args.verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()
        for q, loss in metrics["losses"].items():
            logging.info("loss at {:0.3g}: {:0.3g}".format(q, np.mean(loss)))
        with open(os.path.join(TEST, "{}.eval.pkl".format(args.dataset)), "wb") as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

    # Evaluate conditional probability.
    logging.debug("Evaluating on quantiles: {}".format(args.quantiles))
    loss_fn = LossFunction(model)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    losses = {q: [] for q in quantiles}
    masks = {q: [] for q in quantiles}
    metrics = {"args": args, "losses": losses, "masks": masks}
    with interrupt(save, metrics):
        for batch in partition_data(data, args.batch_size):
            full_loss = loss_fn(batch)
            for q in quantiles:
                while True:
                    mask = torch.distributions.Bernoulli(q).sample((len(batch),))
                    if mask.sum() >= 1 and (1 - mask).sum() >= 1:
                        break
                masked_batch = [col if m else None for (col, m) in zip(batch, mask)]
                masked_loss = loss_fn(masked_batch)
                num_imputed_cells = len(batch[0]) * (1 - mask).sum().item()
                loss = (full_loss - masked_loss) / num_imputed_cells
                assert loss > 0
                losses[q].append(loss)
                masks[q].append(mask)
            if args.verbose:
                sys.stdout.write(".")
                sys.stdout.flush()

    save(metrics)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat evaluation")
    parser.add_argument("--dataset", default="boston_housing")
    parser.add_argument("--max-num-rows", default=9999999999, type=int)
    parser.add_argument("-q", "--quantiles", default="0.1,0.2,0.5,0.8,0.9")
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(relativeCreated) 9d %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info("\n".join(
        ["Config:"] +
        ["\t{} = {}".format(key, value)
         for (key, value) in sorted(vars(args).items())]))

    try:
        main(args)
    except (ValueError, RuntimeError, AssertionError) as e:
        print(e)
        import pdb
        pdb.post_mortem(e.__traceback__)
