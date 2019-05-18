from __future__ import absolute_import, division, print_function

import os
import signal
import sys
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWDATA = os.path.join(ROOT, "rawdata")
DATA = os.path.join(ROOT, "data")
RESULTS = os.environ.get("RESULTS", os.path.join(ROOT, "results"))
TRAIN = os.path.join(RESULTS, "train")
TEST = os.path.join(RESULTS, "test")

if not os.path.exists(TRAIN):
    os.makedirs(TRAIN)
if not os.path.exists(TEST):
    os.makedirs(TEST)


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
    except (ValueError, RuntimeError, AssertionError) as e:
        print(e)
        import pdb
        pdb.post_mortem(e.__traceback__)
