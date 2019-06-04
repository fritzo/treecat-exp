from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys

import numpy as np
import pyro
import torch
from pyro.infer import TraceEnum_ELBO
from six.moves import cPickle as pickle

from treecat_exp.corruption import corrupt
from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TEST, TRAIN, interrupt, pdb_post_mortem


class LossFunction(object):
    def __init__(self, model):
        self.model = model
        self.elbo = TraceEnum_ELBO(max_plate_nesting=1)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.elbo.loss(self.model.model, self.model.guide, *args, **kwargs)


def main(args):
    name = "{}.treecat.{}".format(args.dataset, args.capacity)

    # Load data.
    features, data, mask = load_data(args)
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("loaded {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))
    logging.info("\n".join(["Features:"] + [str(f) for f in features]))

    # Load a trained model.
    logging.debug("Loading model")
    pyro.set_rng_seed(args.seed)
    pyro.get_param_store().load(os.path.join(TRAIN, "{}.model.pyro".format(name)))
    with open(os.path.join(TRAIN, "{}.model.pkl".format(name)), "rb") as f:
        model = pickle.load(f)

    def save(metrics):
        if args.verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()
        for q, loss in metrics["losses"].items():
            logging.info("loss at {:0.3g}: {:0.3g}".format(q, np.mean(loss)))
        with open(os.path.join(TEST, "{}.eval.pkl".format(name)), "wb") as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

    # Evaluate conditional probability.
    logging.debug("Evaluating on quantiles: {}".format(args.quantiles))
    loss_fn = LossFunction(model)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    losses = {q: [] for q in quantiles}
    masks = {q: [] for q in quantiles}
    metrics = {"args": args, "losses": losses, "masks": masks}
    with interrupt(save, metrics):
        for batch_data, batch_mask in partition_data(data, mask, args.batch_size):
            full_loss = loss_fn(batch_data, batch_mask)
            for q in quantiles:
                corrupted = corrupt(batch_data, batch_mask, delete_prob=q)
                corrupted_loss = loss_fn(corrupted["data"], corrupted["mask"])
                num_imputed_cells = len(batch_data[0]) * corrupted["delete_mask"].float().sum().item()
                loss = (full_loss - corrupted_loss) / num_imputed_cells
                assert loss > 0
                losses[q].append(loss)
                masks[q].append(corrupted["delete_mask"])
            if args.verbose:
                sys.stdout.write(".")
                sys.stdout.flush()

    save(metrics)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    pyro.enable_validation(__debug__)

    parser = argparse.ArgumentParser(description="TreeCat evaluation")
    parser.add_argument("--dataset", default="housing")
    parser.add_argument("--max-num-rows", default=9999999999, type=int)
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-q", "--quantiles", default="0.1,0.2,0.5,0.8,0.9")
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("-m", "--model", default="treecat")
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(relativeCreated) 9d %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info("\n".join(
        ["Config:"] +
        ["\t{} = {}".format(key, value)
         for (key, value) in sorted(vars(args).items())]))

    with pdb_post_mortem():
        main(args)
