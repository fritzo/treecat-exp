from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import signal
import sys
from contextlib import contextmanager

import numpy as np
from six.moves import cPickle as pickle

import pyro
from pyro.contrib.tabular import TreeCat
from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.regression import Regressor
from treecat_exp.util import TEST, TRAIN


@contextmanager
def interrupt(fn, *args, **kwargs):
    signal.signal(signal.SIGINT, lambda *_: fn(*args, **kwargs))
    yield
    signal.signal(signal.SIGINT, signal.default_int_handler)


def main(args):
    # Load data.
    features, data = load_data(args)
    num_rows = len(data[0])
    num_cells = num_rows * len(features)
    logging.info("loaded {} rows x {} features = {} cells".format(
        num_rows, len(features), num_cells))
    logging.info("\n".join(["Features:"] + [str(f) for f in features]))

    # Load a trained imputation model.
    logging.debug("Loading model")
    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(__debug__)
    pyro.get_param_store().clear()
    pyro.get_param_store().load(os.path.join(TRAIN, "{}.model.pyro".format(args.dataset)))
    model = TreeCat(features, args.capacity)
    model.load()

    # Split data into train and test.
    batches = list(partition_data(data, args.batch_size))
    batches = batches[:args.max_num_batches]
    train_batches = batches[:len(batches) // 2]
    test_batches = batches[len(batches) // 2:]

    # Train a set of regression models.
    logging.debug("Training at quantiles: {}".format(args.quantiles))
    quantiles = [float(q) for q in args.quantiles.split(",")]
    out_feature = model.features[args.feature_to_predict]
    predictors = {q: Regressor(model.features, out_feature, model.impute, q)
                  for q in quantiles}
    for batch in train_batches:
        for predictor in predictors.values():
            predictor.train(batch)
        if args.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

    # Evaluate accuracy of a predictor.
    logging.debug("Evaluating at quantiles: {}".format(args.quantiles))
    scores = {q: [] for q in quantiles}
    for batch in test_batches:
        for q, predictor in predictors.items():
            score = predictor.test(batch)
            scores[q].append(score)
        if args.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

    # Save results.
    for q, score in scores.items():
        logging.info("score at {:0.3g}: {:0.3g}".format(q, np.mean(score)))
    metrics = {"args": args, "scores": scores}
    with open(os.path.join(TEST, "{}.eval_predictor.pkl".format(args.dataset)), "wb") as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat evaluation of prediction")
    parser.add_argument("--dataset", default="boston_housing")
    parser.add_argument("--feature_to_predict", default=0, type=int)
    parser.add_argument("--max-num-rows", default=9999999999, type=int)
    parser.add_argument("--max-num-batches", default=9999999999, type=int)
    parser.add_argument("-c", "--capacity", default=8, type=int)
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
