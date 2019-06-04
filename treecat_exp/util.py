from __future__ import absolute_import, division, print_function

import os
import signal
import sys
from contextlib import contextmanager

import torch
from six.moves import cPickle as pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWDATA = os.path.join(ROOT, "rawdata")
DATA = os.path.join(ROOT, "data")
RESULTS = os.environ.get("RESULTS", os.path.join(ROOT, "results"))
TRAIN = os.path.join(RESULTS, "train")
CLEANUP = os.path.join(RESULTS, "cleanup")
TEST = os.path.join(RESULTS, "test")


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


for path in [RAWDATA, DATA, TRAIN, TEST, CLEANUP]:
    mkdir_p(path)


@contextmanager
def interrupt(fn, *args, **kwargs):
    signal.signal(signal.SIGINT, lambda *_: fn(*args, **kwargs))
    yield
    signal.signal(signal.SIGINT, signal.default_int_handler)


@contextmanager
def pdb_post_mortem():
    if not sys.__stdin__.isatty():
        yield
        return

    try:
        yield
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem(e.__traceback__)


def to_cuda(x):
    """
    Moves Tensors to cuda; returns python objects unmodified.
    """
    if isinstance(x, torch.Tensor):
        return x.cuda()
    if isinstance(x, list):
        return [to_cuda(item) for item in x]
    if x in (None, False, True):
        return x
    raise ValueError(x)


def save_object(data, path):
    with open(path, "wb") as f:
        torch.save(data, f, pickle_module=pickle, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def load_object(path):
    map_location = None if torch.cuda.is_available() else "cpu"
    with open(path, "rb") as f:
        return torch.load(f, map_location=map_location, pickle_module=pickle)
