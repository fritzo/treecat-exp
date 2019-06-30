from __future__ import absolute_import, division, print_function

import argparse

import pyro

from treecat_exp.config import fill_in_defaults
from treecat_exp.preprocess import load_data
from treecat_exp.training import train_treecat
from treecat_exp.util import pdb_post_mortem


def main(args):
    # Load data.
    features, data, mask = load_data(args)
    name = "{}.{}.{}".format(args.dataset, args.model, args.capacity)
    if args.suffix:
        name = "{}.{}".format(name, args.suffix)
    if args.model == "treecat":
        train_treecat(name, features, data, mask, args)
    else:
        raise ValueError("Unknown model: {}".format(args.model))


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    pyro.enable_validation(__debug__)

    parser = argparse.ArgumentParser(description="TreeCat training")
    parser.add_argument("--dataset", default="housing")
    parser.add_argument("-r", "--max-num-rows", default=1000000000, type=int)
    parser.add_argument("-m", "--model", default="treecat")
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("--treecat-method", default="map")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-ar", "--annealing-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-i", "--init-size", default=1000000000, type=int)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--default-config", action="store_true")
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    fill_in_defaults(args)

    with pdb_post_mortem():
        main(args)
