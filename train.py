from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import signal
from contextlib import contextmanager

import torch
from six.moves import cPickle as pickle

import pyro
from pyro.contrib.tabular import TreeCat, TreeCatTrainer
from pyro.contrib.tabular.treecat import print_tree
from pyro.optim import Adam
from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.util import TRAIN


def print_params(model):
    torch.set_printoptions(precision=3, linewidth=120)
    logging.debug("\n".join(
        ["Param store:", "----------------------------------------"] +
        ["{} =\n{}".format(key, value.data.cpu())
         for key, value in sorted(pyro.get_param_store().items())] +
        ["----------------------------------------"]))
    feature_names = [f.name for f in model.features]
    logging.debug("Tree:\n{}".format(print_tree(model.edges, feature_names)))


@contextmanager
def printing_params(model):
    signal.signal(signal.SIGINT, lambda *_: print_params(model))
    yield
    signal.signal(signal.SIGINT, signal.default_int_handler)


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
    features, data = load_data(args)
    if args.only_features:
        fs = [int(f) for f in args.only_features.split(",")]
        features = [features[f] for f in fs]
        data = [data[f] for f in fs]
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("loaded {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))
    logging.debug("\n".join(["Features:"] + [str(f) for f in features]))

    # Initialize the model.
    logging.debug("Initializing from {} rows".format(args.init_size))
    pyro.set_rng_seed(args.seed)
    pyro.get_param_store().clear()
    pyro.enable_validation(__debug__)
    model = TreeCat(features, args.capacity, annealing_rate=args.annealing_rate)
    optim = Adam({"lr": args.learning_rate})
    trainer = TreeCatTrainer(model, optim)
    for batch in partition_data(data, args.init_size):
        trainer.init(batch)
        break
    print_params(model)

    # Train a model.
    logging.debug("Training for {} epochs".format(args.num_epochs))
    tree_monitor = TreeMonitor(model.edges)
    param_store_monitor = ParamStoreMonitor()
    stepsizes = []
    losses = []
    with printing_params(model):
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in partition_data(data, args.batch_size):
                loss = trainer.step(batch, num_rows=num_rows)
                loss /= num_cells
                losses.append(loss)
                epoch_loss += loss
                num_batches += 1

                stepsize = param_store_monitor.get_diffs()
                feature_stepsize = sum(stepsize.values())
                stepsize["tree"] = tree_monitor.get_diff(model.edges)
                stepsizes.append(stepsize)
                logging.debug("tree_stepsize = {:0.4g}, feature_stepsize = {:0.4g}, loss = {:0.4g}".format(
                    stepsize["tree"], feature_stepsize, loss))
            logging.info("epoch {} loss = {}".format(epoch, epoch_loss / num_batches))

            # Save model and metadata.
            model.save()
            pyro.get_param_store().save(os.path.join(TRAIN, "{}.model.pyro".format(args.dataset)))
            meta = {"args": args, "losses": losses, "stepsizes": stepsizes}
            with open(os.path.join(TRAIN, "{}.meta.pkl".format(args.dataset)), "wb") as f:
                pickle.dump(meta, f, pickle.HIGHEST_PROTOCOL)
    print_params(model)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat training")
    parser.add_argument("--dataset", default="boston_housing")
    parser.add_argument("--max-num-rows", default=1000000000, type=int)
    parser.add_argument("--only-features")
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    parser.add_argument("-ar", "--annealing-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-i", "--init-size", default=1024, type=int)
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
