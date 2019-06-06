from __future__ import absolute_import, division, print_function

import argparse
import itertools
import logging
import multiprocessing
import subprocess
from sys import executable as python


def cleanup(args):
    model, dataset, delete_percent, args = args
    command = [
        python,
        "cleanup.py",
        "--model={}".format(model),
        "--dataset={}".format(dataset),
        "--delete-percent={}".format(delete_percent),
    ]
    if args.verbose:
        command.append("--verbose")
    if args.pdb:
        command.append("--pdb")
    if args.log_errors:
        command.append("--log-errors")
    if args.smoketest:
        command.extend([
            "--max-num-rows=50",
            "--batch-size=20",
            "--num-epochs=1",
            "--custom-config",
        ])
    print("#" * 80)
    print("  \\\n".join(command))
    if args.log_errors:
        subprocess.call(command)
    else:
        subprocess.check_call(command)


def main(args):
    experiments = args.experiments.split(",")

    if "cleanup" in experiments:
        logging.info("Running cleanup experiment")
        models = args.models.split(",")
        datasets = args.datasets.split(",")
        delete_percents = [10] if args.smoketest else [10, 20, 33, 50, 67, 80, 90]
        configs = list(itertools.product(models, datasets, delete_percents, [args]))

        if args.parallel:
            multiprocessing.Pool().map(cleanup, configs)
        elif args.opus:
            raise NotImplementedError("TODO(jpchen) make it easy to run this on opus")
            # Note this saves results to the file system.
        else:
            for config in configs:
                cleanup(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiments", help="list of experiments to run",
                        default="cleanup")
    parser.add_argument("--models", help="list of models to train",
                        default="fancy,treecat,vae")
    parser.add_argument("--datasets", help="list of datasets",
                        default="housing,news,credit,census,lending")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--pdb", action="store_true",
                        help="On error, open a debugger")
    parser.add_argument("-e", "--log-errors", action="store_true",
                        help="On error, log to errors.log and continue")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="Run jobs in parallel using multiprocessing")
    parser.add_argument("-o", "--opus", action="store_true",
                        help="Run jobs in parallel on opus")
    parser.add_argument("-t", "--smoketest", action="store_true")
    args = parser.parse_args()
    main(args)
