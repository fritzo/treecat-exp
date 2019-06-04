from __future__ import absolute_import, division, print_function

import argparse
import logging

import pyro

from treecat_exp.preprocess import load_data
from treecat_exp.util import pdb_post_mortem
from treecat_exp.training import train_treecat


def main(args):
    # Load data.
    features, data, mask = load_data(args)
    name = "{}.{}.{}".format(args.dataset, args.model, args.capacity)
    train_treecat(name, features, data, mask, args)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat training")
    parser.add_argument("--dataset", default="boston_housing")
    parser.add_argument("-r", "--max-num-rows", default=1000000000, type=int)
    parser.add_argument("-m", "--model", default="treecat")
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-ar", "--annealing-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-i", "--init-size", default=1000000000, type=int)
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
