from __future__ import absolute_import, division, print_function

import csv
import logging
import os
from collections import defaultdict

import torch
from observations import boston_housing
from pyro.contrib.tabular import Boolean, Discrete, Real
from six.moves import cPickle as pickle

from treecat_exp.util import DATA, RAWDATA


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
        x_train = x_train[torch.randperm(len(x_train))]
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
    num_rows = min(2458285, args.max_num_rows)
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
            for i, row in enumerate(reader):
                if i == num_rows:
                    break
                for j, (cell, support) in enumerate(zip(row[1:], supports)):
                    value = support.setdefault(int(cell), len(support))
                    assert value <= 255
                    data[i, j] = value

        supports = [list(sorted(s)) for s in supports]
        dataset = {
            "header": header,
            "supports": supports,
            "data": data,
            "args": args,
        }
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    features = []
    data = []
    for j, support in enumerate(dataset["supports"]):
        if len(support) >= 2:
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
