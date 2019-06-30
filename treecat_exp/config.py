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
            "learning_rate": 0.03,
            "annealing_rate": 0.02,
            "num_epochs": 200,
        },
        "credit": {
            "capacity": 32,
            "batch_size": 2000,
            "learning_rate": 0.03,
            "annealing_rate": 0.02,
            "num_epochs": 35,
        },
        "news": {
            "capacity": 32,
            "batch_size": 2048,
            "learning_rate": 0.03,
            "annealing_rate": 0.02,
            "num_epochs": 30,
        },
        "census": {
            "capacity": 32,
            "batch_size": 6144,
            "learning_rate": 0.03,
            "annealing_rate": 0.02,
            "num_epochs": 1,
        },
        "lending": {
            "capacity": 32,
            "batch_size": 512,
            "learning_rate": 0.03,
            "annealing_rate": 0.02,
            "num_epochs": 1,
        },
    },
    "gain": {
        "housing": {
            "learning_rate": 0.1,
            "batch_size": 128,
            "num_epochs": 1200,
            "gen_layer_sizes": [180, 64],
            "disc_layer_sizes": [120, 64],
        },
        "credit": {
            "learning_rate": 0.01,
            "batch_size": 1000,
            "num_epochs": 200,
            "gen_layer_sizes": [200, 64],
            "disc_layer_sizes": [120, 64],
        },
        "news": {
            "learning_rate": 0.01,
            "batch_size": 1000,
            "num_epochs": 300,
            "gen_layer_sizes": [200, 64],
            "disc_layer_sizes": [120, 64],
        },
        "census": {
            # untested on full data
            "batch_size": 1000,
            "learning_rate": 0.01,
            "num_epochs": 10,
            "gen_layer_sizes": [200, 64],
            "disc_layer_sizes": [180, 64],
            "max_num_rows": 100000,
        },
        "lending": {
            # untested on full data
            "batch_size": 1000,
            "learning_rate": 0.01,
            "num_epochs": 10,
            "gen_layer_sizes": [200, 64],
            "disc_layer_sizes": [120, 64],
            "max_num_rows": 100000,
        },
    },
    "vae": {
        "housing": {
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 200,
            "encoder_layer_sizes": [120, 64],
            "decoder_layer_sizes": [120, 64],
            "kl_factor": 1e-4,
        },
        "credit": {
            "learning_rate": 0.001,
            "batch_size": 1000,
            "num_epochs": 50,
            "encoder_layer_sizes": [200, 64],
            "decoder_layer_sizes": [200, 64],
            "kl_factor": 1e-4,
        },
        "news": {
            "learning_rate": 0.001,
            "batch_size": 1000,
            "num_epochs": 50,
            "encoder_layer_sizes": [120, 64],
            "decoder_layer_sizes": [120, 64],
            "kl_factor": 1e-4,
        },
        "census": {
            "batch_size": 1000,
            "learning_rate": 0.001,
            "num_epochs": 2,
            "encoder_layer_sizes": [180, 64],
            "decoder_layer_sizes": [180, 64],
            # "max_num_rows": 100000,
            "kl_factor": 1e-4,
        },
        "lending": {
            "batch_size": 1000,
            "learning_rate": 0.001,
            "num_epochs": 2,
            "encoder_layer_sizes": [120, 64],
            "decoder_layer_sizes": [120, 64],
            "kl_factor": 1e-4,
        },
    },
    "vaeiter": {
        "housing": {
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 200,
            "encoder_layer_sizes": [120, 64],
            "decoder_layer_sizes": [120, 64],
            "kl_factor": 1e-4,
            "vae_iters": 10,
        },
        "credit": {
            "learning_rate": 0.001,
            "batch_size": 1000,
            "num_epochs": 50,
            "encoder_layer_sizes": [200, 64],
            "decoder_layer_sizes": [200, 64],
            "max_num_rows": 100000,
            "kl_factor": 1e-4,
            "vae_iters": 10,
        },
        "news": {
            "learning_rate": 0.001,
            "batch_size": 1000,
            "num_epochs": 50,
            "encoder_layer_sizes": [120, 64],
            "decoder_layer_sizes": [120, 64],
            "kl_factor": 1e-4,
            "vae_iters": 10,
        },
        "census": {
            "batch_size": 1000,
            "learning_rate": 0.001,
            "num_epochs": 2,
            "encoder_layer_sizes": [180, 64],
            "decoder_layer_sizes": [180, 64],
            # "max_num_rows": 100000,
            "kl_factor": 1e-4,
            "vae_iters": 10,
        },
        "lending": {
            "batch_size": 1000,
            "learning_rate": 0.001,
            "num_epochs": 2,
            "encoder_layer_sizes": [120, 64],
            "decoder_layer_sizes": [120, 64],
            "kl_factor": 1e-4,
            "vae_iters": 10,
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
