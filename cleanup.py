from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import pyro
import torch
from pyro.contrib.tabular import TreeCat
from pyro.contrib.tabular.features import Real

from treecat_exp.config import fill_in_defaults
from treecat_exp.corruption import corrupt
from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.training import load_treecat, train_treecat
from treecat_exp.util import CLEANUP, TEST, save_object, load_object, pdb_post_mortem, to_cuda


def cleanup(name, features, data, mask, args):
    corrupted_filename = os.path.join(CLEANUP, "{}.corrupted.pkl".format(name))
    cleaned_filename = os.path.join(CLEANUP, "{}.cleaned.pkl".format(name))
    if os.path.exists(cleaned_filename) and os.path.exists(corrupted_filename):
        corrupted = load_object(corrupted_filename)
        cleaned = load_object(cleaned_filename)
        model = load_treecat(name)
        return corrupted, cleaned, model

    # Currupt data.
    logging.debug("Corrupting dataset")
    corrupted = corrupt(data, mask,
                        delete_prob=args.delete_percent / 100.,
                        replace_prob=args.replace_percent / 100.)
    corrupted["args"] = args
    save_object(corrupted, corrupted_filename)

    # Train model on corrupted data.
    # Models should implement methods .sample(data,mask) and .log_prob(data,mask).
    logging.debug("Training model on corrupted data")
    if args.model == "treecat":
        model = train_treecat(name, features, corrupted["data"], corrupted["mask"], args)
    elif args.model == "vae":
        raise NotImplementedError("TODO(jpchen) train a model")
    else:
        raise ValueError("Unknown model: {}".format(args.model))

    # Clean up data using trained model.
    logging.debug("Cleaning up dataset")
    cleaned_data = [torch.empty_like(col) for col in data]
    cleaned_mask = [True] * len(cleaned_data)
    begin = 0
    for batch_data, batch_mask in partition_data(corrupted["data"], corrupted["mask"], args.batch_size):
        if args.cuda:
            batch_data = to_cuda(batch_data)
            batch_mask = to_cuda(batch_mask)
        with torch.no_grad():
            # TODO(jpchen) Ensure all models support a .sample() method for imputation.
            batch_data = model.sample(batch_data, batch_mask)
        end = begin + len(batch_data[0])
        for cleaned_col, batch_col in zip(cleaned_data, batch_data):
            cleaned_col[begin:end] = batch_col.cpu()
        begin = end
    cleaned = {
        "feature": features,
        "data": cleaned_data,
        "mask": cleaned_mask,
        "args": args,
    }
    save_object(cleaned, cleaned_filename)

    return corrupted, cleaned, model


def main(args):
    # Load original data.
    features, data, mask = load_data(args)
    name = "cleanup.{}.{}.{}.{}.{}".format(
        args.delete_percent, args.replace_percent, args.seed, args.dataset, args.model)

    # Corrupt then cleanup data.
    corrupted, cleaned, model = cleanup(name, features, data, mask, args)
    pyro.set_rng_seed(args.seed)

    # Evaluate loss.
    logging.debug("Evaluating loss")
    losses = []
    num_cleaned = []
    for i, (true_col, cleaned_col) in enumerate(zip(data, cleaned["data"])):
        if mask[i] is not True:
            true_col = true_col[mask[i]]
            cleaned_col = cleaned_col[mask[i]]
        if isinstance(features[i], Real):
            loss = (true_col - cleaned_col).pow(2).mean() / true_col.std()
        else:
            loss = (true_col != cleaned_col).float().mean()
        num_cleaned.append((corrupted["mask"][i] != cleaned["mask"][i]).float().sum().item())
        losses.append(loss.item() / max(num_cleaned[-1], 1e-20))
    metrics = {
        "losses": losses,
        "num_cleaned": num_cleaned,
        "num_rows": len(data[0]),
        "num_cols": len(data),
        "args": args,
    }

    # Evaluate posterior predictive likelihood.
    if isinstance(model, TreeCat):
        log_prob = 0.
        true_batches = partition_data(data, mask, args.batch_size)
        corr_batches = partition_data(corrupted["data"], corrupted["mask"], args.batch_size)
        for (true_data, true_mask), (corr_data, corr_mask) in zip(true_batches, corr_batches):
            if args.cuda:
                true_data = to_cuda(true_data)
                true_mask = to_cuda(true_mask)
                corr_data = to_cuda(corr_data)
                corr_mask = to_cuda(corr_mask)
            with torch.no_grad():
                # TODO(jpchen) Ensure all models support a .log_prob() method for density.
                log_prob += (model.log_prob(true_data, true_mask) -
                             model.log_prob(corr_data, corr_mask))
        metrics["posterior_predictive"] = log_prob / sum(num_cleaned)

    logging.debug("Metrics:")
    for key, value in sorted(metrics.items()):
        logging.debug("{} = {}".format(key, value))
    save_object(metrics, os.path.join(TEST, "{}.pkl".format(name)))


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
    pyro.enable_validation(__debug__)

    parser = argparse.ArgumentParser(description="Data cleanup experiment")
    parser.add_argument("--delete-percent", default=50, type=int)
    parser.add_argument("--replace-percent", default=0, type=int)
    parser.add_argument("--dataset", default="housing")
    parser.add_argument("-r", "--max-num-rows", default=1000000000, type=int)
    parser.add_argument("-m", "--model", default="treecat")
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-ar", "--annealing-rate", default=0.01, type=float)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-i", "--init-size", default=1000000000, type=int)
    parser.add_argument('--default-config', dest='default_config', action='store_true')
    parser.add_argument('--custom-config', dest='default_config', action='store_false')
    parser.set_defaults(default_config=True)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    fill_in_defaults(args)

    with pdb_post_mortem():
        main(args)
