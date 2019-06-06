from __future__ import absolute_import, division, print_function

import argparse
import itertools
import logging
import subprocess
from sys import executable as python


def cleanup(model, dataset, delete_percent, args):
    command = [
        python,
        "cleanup.py",
        "--model={}".format(model),
        "--dataset={}".format(dataset),
        "--delete-percent={}".format(delete_percent),
        "--verbose",
    ]
    if args.smoketest:
        command.extend([
            "--max-num-rows=50",
            "--batch-size=20",
            "--num-epochs=1",
            "--custom-config",
        ])
    print("#" * 80)
    print("  \\\n".join(command))
    subprocess.check_call(command)


def main(args):
    experiments = args.experiments.split(",")

    if "cleanup" in experiments:
        logging.info("Running cleanup experiment")
        models = args.models.split(",")
        datasets = args.datasets.split(",")
        delete_percents = [10] if args.smoketest else [10, 20, 33, 50, 67, 80, 90]
        configs = list(itertools.product(models, datasets, delete_percents))

        # TODO(jpchen) make it easy to run this on opus.
        # Note this saves results to the file system.
        for model, dataset, delete_percent in configs:
            cleanup(model, dataset, delete_percent, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiments", help="list of experiments to run",
                        default="cleanup")
#     parser.add_argument("--models", help="list of models to train",
#                         default="treecat,vae")
    parser.add_argument("--models", help="list of models to train",
                        default="treecat")
#     parser.add_argument("--datasets", help="list of datasets",
#                         default="housing,news,census,lending,credit")
    parser.add_argument("--datasets", help="list of datasets",
                        default="credit")
    parser.add_argument("--smoketest", action="store_true")
    args = parser.parse_args()
    main(args)
