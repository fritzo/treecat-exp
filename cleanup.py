from __future__ import absolute_import, division, print_function

import argparse
import datetime
import logging
import os
import re
import sys

import pyro
import torch
from pyro.contrib.tabular import TreeCat
from pyro.contrib.tabular.features import Boolean, Discrete, Real

from treecat_exp.config import fill_in_defaults
from treecat_exp.corruption import corrupt
from treecat_exp.fancy_impute import load_fancy_imputer, train_fancy_imputer
from treecat_exp.preprocess import load_data, partition_data
from treecat_exp.training import load_treecat, train_treecat
from treecat_exp.util import CLEANUP, TEST, diversity, load_object, pdb_post_mortem, save_object, to_cuda
from treecat_exp.vae.vae import load_vae, train_vae
from treecat_exp.gain import load_gain, train_gain


def cleanup(name, features, data, mask, args):
    corrupted_filename = os.path.join(CLEANUP, "{}.corrupted.pkl".format(name))
    cleaned_filename = os.path.join(CLEANUP, "{}.cleaned.pkl".format(name))
    if os.path.exists(cleaned_filename) and os.path.exists(corrupted_filename) and not args.clean:
        corrupted = load_object(corrupted_filename)
        cleaned = load_object(cleaned_filename)
        if args.model.startswith("treecat"):
            model = load_treecat(name)
        elif args.model == "vae":
            model = load_vae(name)
        elif args.model == "gain":
            model = load_gain(name)
        elif args.model.startswith("fancy"):
            model = load_fancy_imputer(name)
        else:
            raise ValueError("Unknown model: {}".format(args.model))
        return corrupted, cleaned, model

    # Corrupt data.
    logging.debug("Corrupting dataset")
    corrupted = corrupt(data, mask,
                        delete_prob=args.delete_percent / 100.,
                        replace_prob=args.replace_percent / 100.)
    corrupted["args"] = args
    save_object(corrupted, corrupted_filename)

    # Train model on corrupted data.
    # Models should implement methods .sample(data,mask) and .log_prob(data,mask).
    logging.debug("Training model on corrupted data")
    if args.model.startswith("treecat"):
        model = train_treecat(name, features, corrupted["data"], corrupted["mask"], args)
    elif args.model == "vae":
        model = train_vae(name, features, corrupted["data"], corrupted["mask"], args)
    elif args.model == "gain":
        model = train_gain(name, features, corrupted["data"], corrupted["mask"], args)
    elif args.model.startswith("fancy"):
        model = train_fancy_imputer(name, features, corrupted["data"], corrupted["mask"], args)
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
            if args.model == "vae":
                batch_data = model.sample(batch_data, batch_mask, iterative=args.iterative)
            else:
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
            loss = (true_col - cleaned_col).pow(2).mean().sqrt() / true_col.std()
        elif isinstance(features[i], (Boolean, Discrete)):
            loss = (true_col != cleaned_col).float().mean() / diversity(true_col)
        else:
            raise ValueError("Unsupported feature type: {}".format(type(features[i])))
        losses.append(loss.item())
        num_cleaned.append((corrupted["mask"][i] != mask[i]).float().sum().item())

    mean_real_loss = torch.tensor([l for i, l in enumerate(losses) if isinstance(features[i], Real)]).mean()
    mean_discrete_loss = torch.tensor([l for i, l in enumerate(losses) if isinstance(features[i], Discrete)]).mean()
    mean_boolean_loss = torch.tensor([l for i, l in enumerate(losses) if isinstance(features[i], Boolean)]).mean()

    metrics = {
        "types": [type(f).__name__ for f in features],
        "losses": losses,
        "mean_real_loss": mean_real_loss,
        "mean_boolean_loss": mean_boolean_loss,
        "mean_discrete_loss": mean_discrete_loss,
        "num_cleaned": num_cleaned,
        "num_rows": len(data[0]),
        "num_cols": len(data),
        "args": args,
    }

    # Evaluate posterior predictive likelihood.
    # Try to ensure all models support a .log_prob() method for density evaluation.
    if hasattr(model, "log_prob"):
        log_probs = []
        true_batches = partition_data(data, mask, args.batch_size)
        corr_batches = partition_data(corrupted["data"], corrupted["mask"], args.batch_size)
        for (true_data, true_mask), (corr_data, corr_mask) in zip(true_batches, corr_batches):
            if args.cuda:
                true_data = to_cuda(true_data)
                true_mask = to_cuda(true_mask)
                corr_data = to_cuda(corr_data)
                corr_mask = to_cuda(corr_mask)
            with torch.no_grad():
                # Compute posterior predictive as conditional probability:
                # log p(imputed | observed) = log p(imputed, observed) - log p(observed)
                log_prob = (model.log_prob(true_data, true_mask) -
                            model.log_prob(corr_data, corr_mask))
            log_probs.append(log_prob.detach().cpu())
        metrics["posterior_predictive"] = torch.cat(log_probs)

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
    parser.add_argument("-m", "--model", default="treecat", help="{treecat, vae, gain, fancy}")
    parser.add_argument('--default-config', dest='default_config', action='store_true')
    parser.add_argument('--custom-config', dest='default_config', action='store_false')
    parser.set_defaults(default_config=True)
    parser.add_argument("--seed", default=123456789, type=int)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--pdb", action="store_true")
    parser.add_argument("--log-errors", action="store_true")
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--clean", action="store_true", default=False,
                        help="whether to run a fresh run (overwrite old results)")

    # Treecat configs
    parser.add_argument("-c", "--capacity", default=8, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-ar", "--annealing-rate", default=0.01, type=float)
    parser.add_argument("-i", "--init-size", default=1000000000, type=int)

    # VAE configs
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--multi", action="store_true", default=False,
                        help="whether to use multi input/output per Camino et al (2018)")
    parser.add_argument("-nlr", "--noise-lr", default=0.001, type=float,
                        help="noise lr (for iterative imputation)")
    parser.add_argument("--encoder-layer-sizes", default=[128, 64], type=list)
    parser.add_argument("--decoder-layer-sizes", default=[128, 64], type=list)
    parser.add_argument("-l", "--logging-interval", default=10, type=int)
    parser.add_argument("--iterative", action="store_true", default=False,
                        help="whether to use iterative imputation")
    parser.add_argument("--tolerance", default=0.001, type=float, help="tolerance for iterative imputation")

    # GAIN configs
    parser.add_argument("--hint", default=0.9, type=float,
                        help="probability of hint (for GAIN)")
    parser.add_argument("--gen-layer-sizes", default=[128, 64], type=list)
    parser.add_argument("--disc-layer-sizes", default=[128, 64], type=list)
    parser.add_argument("--hint-method", default='drop', type=str,
                        help="see comment in treecat_exp.loss.generate_hint")

    # fancy configs
    parser.add_argument("--fancy-method", default="IterativeImputer")
    parser.add_argument("--fancy-n-iter", default=10, type=int)
    parser.add_argument("--fancy-svd-rank", default=10, type=int)
    parser.add_argument("--fancy-knn-neighbors", default=5, type=int)

    args = parser.parse_args()
    fill_in_defaults(args)

    if args.pdb:
        with pdb_post_mortem():
            main(args)
    elif args.log_errors:
        try:
            main(args)
        except Exception as e:
            logging.error("Job failed with error: {}\nSee errors.log".format(e))
            with open("errors.log", "a") as f:
                f.write("# The following command failed at {} with error:\n# {}\npython {}\n\n".format(
                    datetime.datetime.now(),
                    re.sub(r"\s+", " ", str(e)),
                    " \\\n  ".join(a for a in sys.argv if a != "--log-errors")))
            sys.exit(1)
    else:
        main(args)
