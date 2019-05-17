from __future__ import absolute_import, division, print_function

import csv
import logging
import os
from collections import defaultdict

import torch
from observations import boston_housing
from pyro.contrib.tabular import Boolean, Discrete, Real
from six.moves import cPickle as pickle

from treecat_exp.util import DATA, RAWDATA


def load_data(args):
    name = "load_{}".format(args.dataset)
    assert name in globals()
    return globals()[name](args)


def load_boston_housing(args):
    filename = os.path.join(DATA, "boston_housing.pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        x_train, metadata = boston_housing(DATA)
        x_train = x_train[torch.randperm(len(x_train))]
        x_train = torch.tensor(x_train.T, dtype=torch.get_default_dtype()).contiguous()
        features = []
        data = []
        logging.info("loaded {} rows x {} features:".format(x_train.size(1), x_train.size(0)))
        for name, column in zip(metadata["columns"], x_train):
            ftype = Boolean if name == "CHAS" else Real
            features.append(ftype(name))
            data.append(column)
        dataset = {
            "feature": features,
            "data": data,
            "args": args,
        }
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    return dataset["feature"], dataset["data"]


def load_census(args):
    num_rows = min(2458285, args.max_num_rows)
    filename = os.path.join(DATA, "census.{}.pkl".format(num_rows))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        with open(os.path.join(RAWDATA, "uci-us-census-1990", "USCensus1990.data.txt")) as f:
            reader = csv.reader(f)
            header = next(reader)[1:]
            num_cols = len(header)
            supports = [defaultdict(set) for _ in range(num_cols)]
            data = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
            for i, row in enumerate(reader):
                if i == num_rows:
                    break
                for j, (cell, support) in enumerate(zip(row[1:], supports)):
                    value = support.setdefault(int(cell), len(support))
                    assert value <= 255
                    data[i, j] = value

        supports = [list(sorted(s)) for s in supports]
        dataset = {
            "header": header,
            "supports": supports,
            "data": data,
            "args": args,
        }
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    features = []
    data = []
    for j, support in enumerate(dataset["supports"]):
        if len(support) >= 2:
            name = dataset["header"][j]
            features.append(Discrete(name, len(support)))
            data.append(dataset["data"][:, j].long().contiguous())
    return features, data


NEWS_SCHEMA = {
    "n_tokens_title": Real,
    "n_tokens_content": Real,
    "n_unique_tokens": Real,
    "n_non_stop_words": Real,
    "n_non_stop_unique_tokens": Real,
    "num_hrefs": Real,
    "num_self_hrefs": Real,
    "num_imgs": Real,
    "num_videos": Real,
    "average_token_length": Real,
    "num_keywords": Real,
    "data_channel_is_lifestyle": Boolean,
    "data_channel_is_entertainment": Boolean,
    "data_channel_is_bus": Boolean,
    "data_channel_is_socmed": Boolean,
    "data_channel_is_tech": Boolean,
    "data_channel_is_world": Boolean,
    "kw_min_min": Real,
    "kw_max_min": Real,
    "kw_avg_min": Real,
    "kw_min_max": Real,
    "kw_max_max": Real,
    "kw_avg_max": Real,
    "kw_min_avg": Real,
    "kw_max_avg": Real,
    "kw_avg_avg": Real,
    "self_reference_min_shares": Real,
    "self_reference_max_shares": Real,
    "self_reference_avg_sharess": Real,
    "weekday_is_monday": Boolean,
    "weekday_is_tuesday": Boolean,
    "weekday_is_wednesday": Boolean,
    "weekday_is_thursday": Boolean,
    "weekday_is_friday": Boolean,
    "weekday_is_saturday": Boolean,
    "weekday_is_sunday": Boolean,
    "is_weekend": Boolean,
    "LDA_00": Real,
    "LDA_01": Real,
    "LDA_02": Real,
    "LDA_03": Real,
    "LDA_04": Real,
    "global_subjectivity": Real,
    "global_sentiment_polarity": Real,
    "global_rate_positive_words": Real,
    "global_rate_negative_words": Real,
    "rate_positive_words": Real,
    "rate_negative_words": Real,
    "avg_positive_polarity": Real,
    "min_positive_polarity": Real,
    "max_positive_polarity": Real,
    "avg_negative_polarity": Real,
    "min_negative_polarity": Real,
    "max_negative_polarity": Real,
    "title_subjectivity": Real,
    "title_sentiment_polarity": Real,
    "abs_title_subjectivity": Real,
    "abs_title_sentiment_polarity": Real,
}


def load_news(args):
    num_rows = min(39644, args.max_num_rows)
    filename = os.path.join(DATA, "news.{}.pkl".format(num_rows))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        with open(os.path.join(RAWDATA, "uci-online-news-popularity", "OnlineNewsPopularity.csv")) as f:
            reader = csv.reader(f)
            header = [name.strip() for name in next(reader)]
            logging.debug(header)
            num_cols = len(NEWS_SCHEMA)
            data = torch.zeros(num_rows, num_cols, dtype=torch.float)
            names = list(sorted(NEWS_SCHEMA))
            positions = {name: pos for pos, name in enumerate(names)}
            for i, row in enumerate(reader):
                if i == num_rows:
                    break
                for name, cell in zip(header, row):
                    if name in positions:
                        data[i, positions[name]] = float(cell)
        data = data[torch.randperm(len(data))].t().contiguous()
        dataset = {
            "names": names,
            "data": data,
            "args": args,
        }
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    features = []
    data = []
    for name, col in zip(dataset["names"], dataset["data"]):
        features.append(NEWS_SCHEMA[name](name))
        data.append(col)
    return features, data


def partition_data(data, target_size):
    num_rows = len(data[0])
    begin = 0
    while begin < num_rows:
        end = begin + target_size
        yield [col[begin: end] for col in data]
        begin = end
