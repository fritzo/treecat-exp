from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import pyro
import torch
from pyro.contrib.tabular import Mixture, TreeCat
from pyro.contrib.tabular.treecat import print_tree
from pyro.optim import Adam
from six.moves import cPickle as pickle

from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TRAIN, interrupt, pdb_post_mortem


def save(model, meta):
    # Save model and metadata.
    name = "{}.{}.{}".format(args.dataset, args.model, args.capacity)
    pyro.get_param_store().save(os.path.join(TRAIN, "{}.model.pyro".format(name)))
    with open(os.path.join(TRAIN, "{}.model.pkl".format(name)), "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(TRAIN, "{}.meta.pkl".format(name)), "wb") as f:
        pickle.dump(meta, f, pickle.HIGHEST_PROTOCOL)
    if args.verbose:
        torch.set_printoptions(precision=3, linewidth=120)
        logging.debug("\n".join(
            ["Param store:", "----------------------------------------"] +
            ["{} =\n{}".format(key, value.data.cpu())
             for key, value in sorted(pyro.get_param_store().items())] +
            ["----------------------------------------"]))
        feature_names = [f.name for f in model.features]
        if isinstance(model, TreeCat):
            logging.debug("Tree:\n{}".format(print_tree(model.edges, feature_names)))


class TreeMonitor(object):
    def __init__(self, edges):
        self.current = self._random_walk(edges)

    def _random_walk(self, edges, decay=0.75):
        """
        Computes a heuristic distance between two trees.
        """
        V = len(edges) + 1
        p = torch.eye(V)
        p[edges[:, 0], edges[:, 1]] = 1
        p[edges[:, 1], edges[:, 0]] = 1
        return torch.inverse(torch.eye(V) - decay * p / p.sum(0)) * (1 - decay) / V

    def get_diff(self, edges):
        prev = self.current
        self.current = self._random_walk(edges)
        return (self.current - prev).abs().sum().item()


class ParamStoreMonitor(object):
    def __init__(self):
        self.current = {}
        self.get_diffs()

    def get_diffs(self):
        prev = self.current
        self.current = {name: value.unconstrained().data.clone()
                        for name, value in pyro.get_param_store().items()}
        return {name: (curr - prev.get(name, 0)).norm(2).item()
                for name, curr in self.current.items()}


def main(args):
    # Load data.
    features, data, mask = load_data(args)
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("loaded {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))
    logging.debug("\n".join(["Features:"] + [str(f) for f in features]))

    # Initialize the model.
    logging.debug("Initializing {} model from {} rows".format(args.model, args.init_size))
    pyro.set_rng_seed(args.seed)
    pyro.get_param_store().clear()
    pyro.enable_validation(__debug__)
    if args.model == "treecat":
        model = TreeCat(features, args.capacity, annealing_rate=args.annealing_rate)
    elif args.model == "mixture":
        model = Mixture(features, args.capacity)
    else:
        raise ValueError("Unknown model: {}".format(args.model))
    optim = Adam({"lr": args.learning_rate})
    trainer = model.trainer(optim)
    for batch_data, batch_mask in partition_data(data, mask, args.init_size):
        if args.cuda:
            batch_data = [col.cuda() for col in batch_data]
            batch_mask = [col.cuda() if isinstance(col, torch.Tensor) else col
                          for col in batch_mask]
        trainer.init(batch_data, batch_mask)
        break

    # Train a model.
    logging.debug("Training for {} epochs".format(args.num_epochs))
    tree_monitor = None
    if isinstance(model, TreeCat):
        tree_monitor = TreeMonitor(model.edges)
    param_store_monitor = ParamStoreMonitor()
    stepsizes = []
    losses = []
    meta = {"args": args, "losses": losses, "stepsizes": stepsizes}
    with interrupt(save, model, meta):
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            num_batches = 0
            for batch_data, batch_mask in partition_data(data, mask, args.batch_size):
                if args.cuda:
                    batch_data = [col.cuda() for col in batch_data]
                    batch_mask = [col.cuda() if isinstance(col, torch.Tensor) else col
                                  for col in batch_mask]
                loss = trainer.step(batch_data, batch_mask, num_rows=num_rows)
                loss /= num_cells
                losses.append(loss)
                epoch_loss += loss
                num_batches += 1

                stepsize = param_store_monitor.get_diffs()
                feature_stepsize = sum(stepsize.values())
                if tree_monitor is not None:
                    stepsize["tree"] = tree_monitor.get_diff(model.edges)
                    logging.debug("tree_stepsize = {:0.4g}, feature_stepsize = {:0.4g}, loss = {:0.4g}".format(
                        stepsize["tree"], feature_stepsize, loss))
                stepsizes.append(stepsize)
            logging.info("epoch {} loss = {}".format(epoch, epoch_loss / num_batches))
            save(model, meta)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat training")
    parser.add_argument("--dataset", default="boston_housing")
    parser.add_argument("--max-num-rows", default=1000000000, type=int)
    parser.add_argument("-m", "--model", default="treecat")
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-ar", "--annealing-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-i", "--init-size", default=1024, type=int)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("--cuda", action="store_true")
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
