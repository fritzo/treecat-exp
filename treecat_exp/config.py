from __future__ import absolute_import, division, print_function

import copy
import logging

import torch

DEFAULT_CONFIG = {
    # These default values were determined by running train.py
    # and assessing convergence using train.ipynb .
    "treecat": {
        "housing": {
            "capacity": 16,
            "batch_size": 128,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 100,
        },
        "credit": {
            "capacity": 32,
            "batch_size": 2000,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 35,
        },
        "news": {
            "capacity": 32,
            "batch_size": 2048,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 30,
        },
        "census": {
            "capacity": 32,
            "batch_size": 6144,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 1,
        },
        "lending": {
            "capacity": 32,
            "batch_size": 512,
            "learning_rate": 0.02,
            "annealing_rate": 0.02,
            "num_epochs": 1,
        },
    },
    "vae": {
        "housing": {
            "learning_rate": 0.001,
            "num_epochs": 100,
        },
        "credit": {
            "learning_rate": 0.001,
            "batch_size": 2000,
            "num_epochs": 35,
        },
        "news": {
            "learning_rate": 0.001,
            "batch_size": 2045,
            "num_epochs": 30,
        },
        "census": {
            "batch_size": 8192,
            "learning_rate": 0.001,
            "num_epochs": 2,
        },
        "lending": {
            "batch_size": 1440,
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
    "fancyii": {
        "housing": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_method": "IterativeImputer",
        },
        "credit": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_method": "IterativeImputer",
        },
        "news": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_method": "IterativeImputer",
        },
        "census": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_method": "IterativeImputer",
        },
        "lending": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_method": "IterativeImputer",
        },
    },
    "fancysvd": {
        "housing": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_svd_rank": 10,
            "fancy_method": "IterativeSVD",
        },
        "credit": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_svd_rank": 10,
            "fancy_method": "IterativeSVD",
        },
        "news": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_svd_rank": 10,
            "fancy_method": "IterativeSVD",
        },
        "census": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_svd_rank": 10,
            "fancy_method": "IterativeSVD",
        },
        "lending": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_n_iter": 10,
            "fancy_svd_rank": 10,
            "fancy_method": "IterativeSVD",
        },
    },
    "fancyknn": {
        "housing": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_knn_neighbors": 5,
            "fancy_method": "KNN",
        },
        "credit": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_knn_neighbors": 5,
            "fancy_method": "KNN",
        },
        "news": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_knn_neighbors": 5,
            "fancy_method": "KNN",
        },
        "census": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_knn_neighbors": 5,
            "fancy_method": "KNN",
        },
        "lending": {
            "batch_size": 50000,
            "cuda": False,
            "fancy_knn_neighbors": 5,
            "fancy_method": "KNN",
        },
    },
}


# Add treecat variants with fixed capacity.
def _():
    for capacity in [8, 16, 32, 64]:
        configs = copy.deepcopy(DEFAULT_CONFIG["treecat"])
        for config in configs.values():
            config["capacity"] = capacity
        DEFAULT_CONFIG["treecat{}".format(capacity)] = configs


_()


def fill_in_defaults(args):
    """
    Fills in default values defined in the global DEFAULT_CONFIG dict.
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
