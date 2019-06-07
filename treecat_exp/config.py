from __future__ import absolute_import, division, print_function

import torch
import logging

DEFAULT_CONFIG = {
    # These default values were determined by running train.py
    # and assessing convergence using train.ipynb .
    "treecat": {
        "housing": {
            "capacity": 8,
            "batch_size": 128,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 100,
        },
        "credit": {
            "capacity": 8,
            "batch_size": 2000,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 35,
        },
        "news": {
            "capacity": 8,
            "batch_size": 2048,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 30,
        },
        "census": {
            "capacity": 8,
            "batch_size": 8192,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 1,
        },
        "lending": {
            "capacity": 16,
            "batch_size": 1440,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 1,
        },
    },
    "vae": {
        "housing": {
        },
        "credit": {
            "batch_size": 2000,
            "learning_rate": 0.001,
            "num_epochs": 5,
        },
        "news": {
        },
        "census": {
        },
        "lending": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "num_epochs": 2,
        },
    },
    "vae_iter_impute": {
        "housing": {
        },
        "news": {
        },
        "census": {
        },
        "lending": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "noise_lr": 0.001,
            "num_epochs": 2,
        },
        "credit": {
        },
    },
    "fancy": {
        "housing": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
        },
        "credit": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
        },
        "news": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
        },
        "census": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
        },
        "lending": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
        },
    },
}


def fill_in_defaults(args):
    """
    Fills in default values defined in the global DEFALT_CONFIG dict.
    """
    if args.default_config:
        if torch.cuda.is_available():
            args.cuda = True
        default_config = DEFAULT_CONFIG[args.model][args.dataset]
        for key, value in default_config.items():
            assert hasattr(args, key)
            assert type(getattr(args, key)) == type(value)
            setattr(args, key, value)

    logging.basicConfig(format="%(relativeCreated) 9d %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info("\n".join(
        ["Config:"] +
        ["\t{} = {}".format(key, value)
         for (key, value) in sorted(vars(args).items())]))
