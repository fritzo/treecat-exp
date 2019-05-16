from __future__ import absolute_import, division, print_function

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWDATA = os.path.join(ROOT, "rawdata")
DATA = os.path.join(ROOT, "data")
TRAIN = os.path.join(ROOT, "results", "train")
TEST = os.path.join(ROOT, "results", "test")

if not os.path.exists(TRAIN):
    os.makedirs(TRAIN)
if not os.path.exists(TEST):
    os.makedirs(TEST)
