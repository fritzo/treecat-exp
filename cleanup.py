from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import pyro
from pyro.contrib.tabular.features import Real
from six.moves import cPickle as pickle

from treecat_exp.config import fill_in_defaults
from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.training import train_treecat
from treecat_exp.util import CLEANUP, TEST, pdb_post_mortem, to_cuda
from treecat_exp.corruption import corrupt
import torch


def cleanup(name, features, data, mask, args):
    cache_filename = os.path.join(CLEANUP, "{}.pkl".format(name))
    if os.path.exists(cache_filename):
        with open(cache_filename, "wb") as f:
            return pickle.load(f)

    # Currupt data.
    logging.debug("Corrupting dataset")
    corrupted = corrupt(data, mask,
                        delete_prob=args.delete_percent / 100.,
                        replace_prob=args.replace_percent / 100.)

    # Train model on corrupted data.
    logging.debug("Training model on corrupted data")
    model = train_treecat(name, features, corrupted["data"], corrupted["mask"], args)

    # Use trained model to clean up data.
    logging.debug("Cleaning up dataset")
    cleaned_data = [torch.empty_like(col) for col in data]
    cleaned_mask = [True] * len(cleaned_data)
    begin = 0
    for batch_data, batch_mask in partition_data(data, mask, args.batch_size):
        if args.cuda:
            batch_data = to_cuda(batch_data)
            batch_mask = to_cuda(batch_mask)
        batch_data = model.sample(batch_data, batch_mask)
        end = begin + len(batch_data[0])
        for cleaned_col, batch_col in zip(cleaned_data, batch_data):
            cleaned_col[begin:end] = batch_col
        begin = end

    # Save cleaned data.
    dataset = {
        "feature": features,
        "data": cleaned_data,
        "mask": cleaned_mask,
        "args": args,
    }
    with open(cache_filename, "wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    return dataset


def main(args):
    # Load original data.
    features, data, mask = load_data(args)
    name = "cleanup.{}.{}.{}.{}.{}".format(
        args.delete_percent, args.replace_percent, args.seed, args.dataset, args.model)

    # Corrupt then cleanup data.
    cleaned = cleanup(name, features, data, mask, args)

    # Evaluate loss.
    logging.debug("Evaluating loss")
    losses = []
    for i, (true_col, cleaned_col) in enumerate(zip(data, cleaned["data"])):
        if mask[i] is not True:
            true_col = true_col[mask[i]]
            cleaned_col = cleaned_col[mask[i]]
        if isinstance(features[i], Real):
            loss = (true_col - cleaned_col).pow(2).mean() / true_col.std()
        else:
            loss = (true_col != cleaned_col).float().mean()
        losses.append(loss.item())
    metrics = {"losses": losses, "args": args}
    with open(os.path.join(TEST, "{}.pkl".format(name)), "wb") as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    assert pyro.__version__ >= "0.3.3"
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
