from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import signal
from collections import defaultdict
from contextlib import contextmanager

import torch
from observations import boston_housing
from six.moves import cPickle as pickle

import pyro
from pyro.contrib.tabular import Boolean, Discrete, Real, TreeCat, TreeCatTrainer
from pyro.contrib.tabular.treecat import print_tree
from pyro.optim import Adam

ROOT = os.path.dirname(os.path.abspath(__file__))
RAWDATA = os.path.join(ROOT, "rawdata")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "data")


def load_data(args):
    name = "load_{}".format(args.dataset)
    assert name in globals()
    return globals()[name](args)


def load_boston_housing(args):
    filename = os.path.join(DATA, "boston_housing.pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        x_train, metadata = boston_housing(DATA)
        x_train = torch.tensor(x_train.T, dtype=torch.get_default_dtype()).contiguous()
        features = []
        data = []
        logging.info("loaded {} rows x {} features:".format(x_train.size(1), x_train.size(0)))
        for name, column in zip(metadata["columns"], x_train):
            ftype = Boolean if name == "CHAS" else Real
            features.append(ftype(name))
            data.append(column)
        dataset = {
            "feature": features,
            "data": data,
            "args": args,
        }
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    return dataset["feature"], dataset["data"]


def load_census(args):
    num_rows = min(2458286, args.max_num_rows)
    filename = os.path.join(DATA, "census.{}.pkl".format(num_rows))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        with open(os.path.join(RAWDATA, "uci-us-census-1990", "USCensus1990.data.txt")) as f:
            reader = csv.reader(f)
            header = next(reader)[1:]
            num_cols = len(header)
            supports = [defaultdict(set) for _ in range(num_cols)]
            data = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
            mask = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
            for i, row in enumerate(reader):
                if i == num_rows:
                    break
                for j, (cell, support) in enumerate(zip(row[1:], supports)):
                    if cell:
                        value = support.setdefault(cell, len(support))
                        assert value <= 255
                        data[i, j] = value
                        mask[i, j] = 1

        supports = [list(sorted(s)) for s in supports]
        dataset = {
            "header": header,
            "supports": supports,
            "data": data,
            "mask": mask,
            "args": args,
        }
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    features = []
    data = []
    for j, support in enumerate(dataset["supports"]):
        if len(support) >= 2 and dataset["mask"][:, j].all():
            name = dataset["header"][j]
            features.append(Discrete(name, len(support)))
            data.append(dataset["data"][:, j].long().contiguous())
    return features, data


def partition_data(data, target_size):
    num_rows = len(data[0])
    begin = 0
    while begin < num_rows:
        end = begin + target_size
        yield [col[begin: end] for col in data]
        begin = end


def print_params(model):
    torch.set_printoptions(precision=3, linewidth=120)
    logging.info("\n".join(
        ["Param store:", "----------------------------------------"] +
        ["{} =\n{}".format(key, value.data.cpu())
         for key, value in sorted(pyro.get_param_store().items())] +
        ["----------------------------------------"]))
    feature_names = [f.name for f in model.features]
    logging.info("Tree:\n{}".format(print_tree(model.edges, feature_names)))


@contextmanager
def printing_params(model):
    signal.signal(signal.SIGINT, lambda *_: print_params(model))
    yield
    signal.signal(signal.SIGINT, signal.default_int_handler)


def _random_walk(edges, decay=0.75):
    V = len(edges) + 1
    p = torch.eye(V)
    p[edges[:, 0], edges[:, 1]] = 1
    p[edges[:, 1], edges[:, 0]] = 1
    return torch.inverse(torch.eye(V) - decay * p / p.sum(0)) * (1 - decay) / V


def tree_distance(old_edges, new_edges):
    old_dist = _random_walk(old_edges)
    new_dist = _random_walk(new_edges)
    return (old_dist - new_dist).abs().sum().item()


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
    logging.info("\n".join(["Features:"] + [str(f) for f in features]))

    # Train a model.
    pyro.set_rng_seed(123456789)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    model = TreeCat(features, args.capacity)
    optim = Adam({"lr": args.learning_rate})
    trainer = TreeCatTrainer(model, optim, backend=args.backend)
    for batch in partition_data(data, args.init_size):
        trainer.init(data)
        break
    print_params(model)
    edges = model.edges
    losses = []
    jumps = []
    with printing_params(model):
        for epoch in range(args.num_epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in partition_data(data, args.batch_size):
                loss = trainer.step(batch, num_rows=num_rows)
                loss /= num_cells
                losses.append(loss)
                jump = tree_distance(edges, model.edges)
                edges = model.edges
                jumps.append(jump)
                logging.debug("  jump = {:0.3g}\tloss = {:0.3g}".format(jump, loss))
                epoch_loss += loss
                num_batches += 1
            logging.info("epoch {} loss = {}".format(epoch, epoch_loss / num_batches))

    # Save model and metadata.
    print_params(model)
    pyro.get_param_store().save(os.path.join(RESULTS, "model.pyro"))
    meta = {
        "args": args,
        "losses": losses,
        "jumps": jumps,
    }
    with open(os.path.join(RESULTS, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat experiments")
    parser.add_argument("--dataset", default="boston_housing")
    parser.add_argument("--max-num-rows", default=1000000000, type=int)
    parser.add_argument("--only-features")
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-i", "--init-size", default=100, type=int)
    parser.add_argument("--backend", default="cpp")
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
        import pdb
        pdb.post_mortem(e.__traceback__)
