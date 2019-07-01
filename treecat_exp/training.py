from __future__ import absolute_import, division, print_function

import logging
import os

import pyro
import torch
from pyro.contrib.tabular import TreeCat
from pyro.contrib.tabular.treecat import print_tree
from pyro.optim import Adam

from treecat_exp.preprocess import partition_data
from treecat_exp.util import TRAIN, interrupt, load_object, save_object, to_cuda


def save_treecat(name, model, meta, args):
    """
    Save model and metadata.
    """
    pyro.get_param_store().save(os.path.join(TRAIN, "{}.model.pyro".format(name)))
    save_object(model, os.path.join(TRAIN, "{}.model.pkl".format(name)))
    save_object(meta, os.path.join(TRAIN, "{}.meta.pkl".format(name)))
    if args.verbose:
        if False:
            torch.set_printoptions(precision=3, linewidth=120)
            logging.debug("\n".join(
                ["Param store:", "----------------------------------------"] +
                ["{} =\n{}".format(key, value.data.cpu())
                 for key, value in sorted(pyro.get_param_store().items())] +
                ["----------------------------------------"]))
        feature_names = [f.name for f in model.features]
        if isinstance(model, TreeCat):
            logging.debug("Tree:\n{}".format(print_tree(model.edges, feature_names)))


def load_treecat(name):
    """
    Load model.
    """
    map_location = None if torch.cuda.is_available() else "cpu"
    pyro.get_param_store().load(os.path.join(TRAIN, "{}.model.pyro".format(name)),
                                map_location=map_location)
    model = load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))
    return model


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


def train_treecat(name, features, data, mask, args):
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("Training on {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))
    logging.debug("\n".join(["Features:"] + [str(f) for f in features]))

    # Initialize the model.
    init_size = min(num_rows, args.init_size)
    logging.debug("Initializing {} model from {} rows".format(args.model, init_size))
    pyro.set_rng_seed(args.seed)
    pyro.get_param_store().clear()
    if args.model.startswith("treecat"):
        model = TreeCat(features, args.capacity, annealing_rate=args.annealing_rate)
    else:
        raise ValueError("Unknown model: {}".format(args.model))
    options = {}
    if args.treecat_method == "map":
        options["optim"] = Adam({"lr": args.learning_rate, "betas": (0.5, 0.9)})
    trainer = model.trainer(args.treecat_method, **options)
    for batch_data, batch_mask in partition_data(data, mask, init_size):
        if isinstance(batch_mask, torch.Tensor):
            if batch_mask.all():
                batch_mask = True
            elif not batch_mask.any():
                batch_mask = False
        if args.cuda:
            batch_data = to_cuda(batch_data)
            batch_mask = to_cuda(batch_mask)
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
    memsizes = []
    meta = {"args": args, "losses": losses, "stepsizes": stepsizes, "memsizes": memsizes}
    with interrupt(save_treecat, name, model, meta, args):
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            num_batches = 0
            for batch_data, batch_mask in partition_data(data, mask, args.batch_size):
                if args.cuda:
                    batch_data = to_cuda(batch_data)
                    batch_mask = to_cuda(batch_mask)
                loss = trainer.step(batch_data, batch_mask, num_rows=num_rows)
                loss /= num_cells
                losses.append(loss)
                epoch_loss += loss
                num_batches += 1

                stepsize = param_store_monitor.get_diffs()
                feature_stepsize = sum(stepsize.values())
                if tree_monitor is not None:
                    stepsize["tree"] = tree_monitor.get_diff(model.edges)
                    logging.debug("tree_stepsize = {:0.4g}, feature_stepsize = {:0.4g}, loss = {:0.4g}"
                                  .format(stepsize["tree"], feature_stepsize, loss))
                stepsizes.append(stepsize)
                memsizes.append(model._feature_model._count_stats)
            logging.info("epoch {} loss = {}".format(epoch, epoch_loss / num_batches))
            save_treecat(name, model, meta, args)

    return model
