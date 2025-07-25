from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys

import numpy as np
import pyro

from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.regression import Regressor
from treecat_exp.util import TEST, TRAIN, load_object, pdb_post_mortem, save_object


def main(args):
    name = "{}.{}".format(args.dataset, args.capacity)

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
    pyro.get_param_store().load(os.path.join(TRAIN, "{}.model.pyro".format(name)))
    model = load_object(os.path.join(TRAIN, "{}.model.pkl".format(name)))

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
    metrics = {"args": args, "scores": scores, "predictors": predictors}
    save_object(metrics, os.path.join(TEST, "{}.eval_predictor.pkl".format(name)))


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    parser = argparse.ArgumentParser(description="TreeCat evaluation of prediction")
    parser.add_argument("--dataset", default="housing")
    parser.add_argument("--feature_to_predict", default=0, type=int)
    parser.add_argument("--max-num-rows", default=9999999999, type=int)
    parser.add_argument("--max-num-batches", default=9999999999, type=int)
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-q", "--quantiles", default="0.1,0.2,0.5,0.8,0.9,1.0")
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

    with pdb_post_mortem():
        main(args)
