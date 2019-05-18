from __future__ import absolute_import, division, print_function

import os
import signal
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
