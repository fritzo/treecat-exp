from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
from collections import defaultdict

import torch
from observations import boston_housing
from six.moves import cPickle as pickle

import pyro
from pyro.contrib.tabular import Boolean, Real, TreeCat, TreeCatTrainer
from pyro.optim import Adam

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

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
            logging.info(" {} {}".format(name, ftype.__name__))
        dataset = {"feature": features, "data": data}
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    return dataset["feature"], dataset["data"]


def load_census(args):
    filename = os.path.join(DATA, "census.pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        num_rows = 2458286
        with open(os.path.join(RAWDATA, "uci-us-census-1990", "USCensus1990.data.txt")) as f:
            reader = csv.reader(f)
            header = next(reader)[1:]
            num_cols = len(header)
            supports = [defaultdict(set) for _ in range(num_cols)]
            data = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
            mask = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
            for i, row in enumerate(reader):
                for j, (cell, support) in enumerate(zip(row[1:], supports)):
                    if cell:
                        value = support.setdefault(cell, len(support))
                        assert value <= 255
                        data[i, j] = value
                        mask[i, j] = 1
        dataset = {
            "header": header,
            "data": data,
            "mask": mask,
            "args": args,
        }
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    raise NotImplementedError("TODO")


def partition_data(data, target_size):
    num_rows = len(data[0])
    begin = 0
    while begin < num_rows:
        end = begin + target_size
        yield [col[begin: end] for col in data]
        begin = end


def main(args):
    # Load data.
    features, data = load_data(args)
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("loaded {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))

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
    losses = []
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for batch in partition_data(data, args.batch_size):
            loss = trainer.step(batch, num_rows=num_rows)
            loss /= num_cells
            losses.append(loss)
            logging.info("  loss = {}".format(loss))
            epoch_loss += loss
        logging.info("epoch {} loss = {}".format(epoch, epoch_loss))

    # Save model and metadata.
    pyro.get_param_store().save(os.path.join(RESULTS, "model.pyro"))
    meta = {
        "args": args,
        "losses": losses,
    }
    with open(os.path.join(RESULTS, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat experiments")
    parser.add_argument("--dataset", default="boston_housing")
    parser.add_argument("-c", "--capacity", default=16, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=22, type=int)
    parser.add_argument("-i", "--init-size", default=100, type=int)
    parser.add_argument("--backend", default="python")
    args = parser.parse_args()
    logging.info("\n".join(
        ["Config:"] +
        ["\t{} = {}".format(key, value)
         for (key, value) in sorted(vars(args).items())]))
    main(args)
