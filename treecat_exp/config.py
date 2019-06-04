from __future__ import absolute_import, division, print_function

import torch

DEFAULT_CONFIG = {
    "treecat": {
        "housing": {
            "capacity": 8,
            "batch_size": 128,
            "learning_rate": 0.01,
            "annealing_rate": 0.01,
            "num_epochs": 100,
        },
        "news": {
            "capacity": 8,
            "batch_size": 2048,
            "learning_rate": 0.01,
            "annealing_rate": 0.01,
            "num_epochs": 10,
        },
        "census": {
            "capacity": 8,
            "batch_size": 8192,
            "learning_rate": 0.01,
            "annealing_rate": 0.01,
            "num_epochs": 2,
        },
        "lending": {
            "capacity": 16,
            "batch_size": 1440,
            "learning_rate": 0.01,
            "annealing_rate": 0.01,
            "num_epochs": 2,
        },
    },
    # TODO add default configs for MIVAE, GAIN, etc.
}


def fill_in_defaults(args):
    """
    Fills in default values defined in the global DEFALT_CONFIG dict.
    """
    if args.default_config:
        default_config = DEFAULT_CONFIG[args.model][args.dataset]
        for key, value in default_config.items():
            assert hasattr(args, key)
            assert type(getattr(args, key)) == type(value)
            setattr(args, key, value)
        if torch.cuda.is_available():
            args.cuda = True
