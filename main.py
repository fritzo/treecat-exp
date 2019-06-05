from __future__ import absolute_import, division, print_function

import argparse
import itertools
from subprocess import check_call
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
    check_call(command)


def main(args):
    # Run cleanup experiment.
    models = ["treecat", "vae"]  # TODO(jpchen) add vae etc.
    models = ["vae"]  # TODO(jpchen) add vae etc.
    datasets = ["housing", "news", "census", "lending"]
    delete_percents = [10] if args.smoketest else [10, 20, 33, 50, 67, 80, 90]
    configs = list(itertools.product(models, datasets, delete_percents))
    # TODO(jpchen) make it easy to run this on opus.
    for model, dataset, delete_percent in configs:
        cleanup(model, dataset, delete_percent, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main experiment runner")
    parser.add_argument("--smoketest", action="store_true")
    args = parser.parse_args()
    main(args)
