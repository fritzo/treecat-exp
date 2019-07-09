from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import subprocess
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from pyro.contrib.examples import boston_housing
from pyro.contrib.tabular import Boolean, Discrete, Real
from six.moves import urllib

from treecat_exp.util import DATA, RAWDATA, load_object, mkdir_p, save_object

Count = Discrete  # We currently don't handle count data.
Text = None  # We currently don't handle text data.


def load_data(args):
    dataset = args.dataset
    if "." in dataset:
        dataset, max_num_rows = dataset.split(".")
        args.max_num_rows = int(max_num_rows)
    name = "load_{}".format(dataset)
    assert name in globals()
    features, data, mask = globals()[name](args)
    args.max_num_rows = data[0].size(0)
    return features, data, mask


def load_housing(args):
    """
    See any of the following:
    http://lib.stat.cmu.edu/datasets/boston
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
    https://github.com/edwardlib/observations/blob/master/observations/boston_housing.py
    """
    # Convert to torch.
    num_rows = min(506, args.max_num_rows)
    cache_filename = os.path.join(DATA, "housing.{}.pkl".format(num_rows))
    if os.path.exists(cache_filename):
        dataset = load_object(cache_filename)
    else:
        x_train, header = boston_housing.load(DATA)
        x_train = x_train[torch.randperm(len(x_train))]
        x_train = x_train[:num_rows]
        x_train = x_train.t().to(dtype=torch.get_default_dtype()).contiguous()
        features = []
        data = []
        logging.info("loaded {} rows x {} features:".format(x_train.size(1), x_train.size(0)))
        for name, column in zip(header, x_train):
            ftype = Boolean if name == "CHAS" else Real
            features.append(ftype(name))
            data.append(column)
        dataset = {
            "features": features,
            "data": data,
            "args": args,
        }
        save_object(dataset, cache_filename)

    # Format columns.
    mask = [True] * len(dataset["data"])
    return dataset["features"], dataset["data"], mask


def load_census(args):
    """
    See https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
    """
    # Convert to torch.
    num_rows = min(2458285, args.max_num_rows)
    cache_filename = os.path.join(DATA, "census.{}.pkl".format(num_rows))
    if os.path.exists(cache_filename):
        dataset = load_object(cache_filename)
    else:
        raw_filename = os.path.join(RAWDATA, "uci-us-census-1990", "USCensus1990.data.txt")
        if not os.path.exists(raw_filename):
            mkdir_p(os.path.dirname(raw_filename))
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt",
                raw_filename)
        with open(raw_filename) as f:
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
        save_object(dataset, cache_filename)

    # Format columns.
    features = []
    data = []
    mask = []
    for j, support in enumerate(dataset["supports"]):
        if len(support) >= 2:
            name = dataset["header"][j]
            features.append(Discrete(name, len(support)))
            data.append(dataset["data"][:, j].long().contiguous())
            mask.append(True)
    return features, data, mask


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
    """
    See https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
    """
    # Convert to torch.
    num_rows = min(39644, args.max_num_rows)
    cache_filename = os.path.join(DATA, "news.{}.pkl".format(num_rows))
    if os.path.exists(cache_filename):
        dataset = load_object(cache_filename)
    else:
        raw_dir = os.path.join(RAWDATA, "uci-online-news-popularity")
        raw_filename = os.path.join(raw_dir, "OnlineNewsPopularity.csv")
        if not os.path.exists(raw_filename):
            logging.info("Downloading online news popularity dataset")
            mkdir_p(raw_dir)
            zip_filename = os.path.join(raw_dir, "OnlineNewsPopularity.zip")
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip",
                zip_filename)
            subprocess.check_call(["unzip", "-o", zip_filename, "-d", raw_dir])
            os.rename(os.path.join(raw_dir, "OnlineNewsPopularity", "OnlineNewsPopularity.csv"),
                      os.path.join(raw_dir, "OnlineNewsPopularity.csv"))

        with open(raw_filename) as f:
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
        save_object(dataset, cache_filename)

    # Format columns.
    features = []
    data = []
    mask = []
    for name, col in zip(dataset["names"], dataset["data"]):
        features.append(NEWS_SCHEMA[name](name))
        data.append(col)
        mask.append(True)
    return features, data, mask


CREDIT_SCHEMA = {
    "LIMIT_BAL": Real,
    "SEX": Discrete,
    "EDUCATION": Discrete,
    "MARRIAGE": Discrete,
    "AGE": Real,
    "PAY_0": Discrete,
    "PAY_2": Discrete,
    "PAY_3": Discrete,
    "PAY_4": Discrete,
    "PAY_5": Discrete,
    "PAY_6": Discrete,
    "BILL_AMT1": Real,
    "BILL_AMT2": Real,
    "BILL_AMT3": Real,
    "BILL_AMT4": Real,
    "BILL_AMT5": Real,
    "BILL_AMT6": Real,
    "PAY_AMT1": Real,
    "PAY_AMT2": Real,
    "PAY_AMT3": Real,
    "PAY_AMT4": Real,
    "PAY_AMT5": Real,
    "PAY_AMT6": Real,
}


def load_credit(args):
    """
    See https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    """
    import xlrd
    import wget
    # Convert to torch.
    num_rows = min(30000, args.max_num_rows)
    cache_filename = os.path.join(DATA, "credit.{}.pkl".format(num_rows))
    if os.path.exists(cache_filename):
        dataset = load_object(cache_filename)
    else:
        raw_dir = os.path.join(RAWDATA, "uci-default-credit-card")
        raw_filename = os.path.join(raw_dir, "DefaultCreditCard.csv")
        if not os.path.exists(raw_filename):
            logging.info("Downloading default credit card dataset")
            mkdir_p(raw_dir)
            wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/" +
                          "00350/default of credit card clients.xls",
                          raw_dir + '/default_credit_card.xls'
                          )
            xl_file = xlrd.open_workbook(raw_dir + "/default_credit_card.xls")
            sheet = xl_file.sheet_by_name("Data")

            # convert xl file to csv
            with open(raw_filename, "w") as csv_file:
                writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                for row_number in range(1, sheet.nrows):  # ignore first row
                    writer.writerow(sheet.row_values(row_number))

        names = list(sorted(CREDIT_SCHEMA))
        num_cols = len(CREDIT_SCHEMA)
        data = torch.zeros(num_rows, num_cols, dtype=torch.float)
        positions = {name: pos for pos, name in enumerate(names)}
        supports = {name: defaultdict(set)
                    for name in names if CREDIT_SCHEMA[name] is Discrete}
        with open(raw_filename) as f:
            reader = csv.reader(f)
            header = [name.replace("\"", "").strip() for name in next(reader)]
            logging.debug(header)
            assert set(names) <= set(header), "invalid schema"
            js = [positions.get(name) for name in header]
            types = [CREDIT_SCHEMA.get(name) for name in header]
            supps = [supports.get(name) for name in header]
            for i, row in enumerate(reader):
                if i == num_rows:
                    break
                for name, cell, typ, support, j in zip(header, row, types, supps, js):
                    cell = float(cell)
                    if typ is None or not cell:
                        continue
                    if typ is Real:
                        value = cell
                    else:
                        value = support.setdefault(cell, len(support))
                    data[i, j] = value
        logging.debug("\n".join(
            ["Cardinalities:"] +
            ["{: >10} {}".format(len(support), name)
             for name, support in sorted(supports.items())]))

        data = data[torch.randperm(len(data))].t().contiguous()
        dataset = {
            "names": names,
            "data": data,
            "supports": supports,
            "args": args,
        }
        save_object(dataset, cache_filename)

    # Format columns.
    features = []
    data = []
    for name, col in zip(dataset["names"], dataset["data"]):
        typ = CREDIT_SCHEMA[name]
        if typ is Discrete:
            cardinality = len(dataset["supports"][name])
            if cardinality == 1:
                logging.debug("Dropping trivial feature: {}".format(name))
                continue
            feature = typ(name, cardinality)
        else:
            feature = typ(name)
        features.append(feature)
        data.append(col.to(feature.dtype))
    mask = [True] * len(features)
    return features, data, mask


LENDING_SCHEMA = {
    "acc_now_delinq": Discrete,
    "acc_open_past_24mths": Discrete,
    "addr_state": Discrete,
    "all_util": Real,
    "annual_inc": Real,
    "annual_inc_joint": Real,
    "application_type": Discrete,
    "avg_cur_bal": Real,
    "bc_open_to_buy": Real,
    "bc_util": Real,
    "chargeoff_within_12_mths": Discrete,
    "collection_recovery_fee": Real,
    "collections_12_mths_ex_med": Discrete,
    "debt_settlement_flag": Discrete,
    "debt_settlement_flag_date": None,
    "deferral_term": None,
    "delinq_2yrs": Real,
    "delinq_amnt": Real,
    "desc": None,
    "disbursement_method": Discrete,
    "dti": Real,
    "dti_joint": Real,
    "earliest_cr_line": None,
    "emp_length": Discrete,
    "emp_title": Text,
    "funded_amnt": Real,
    "funded_amnt_inv": Real,
    "grade": Discrete,
    "hardship_amount": Real,
    "hardship_dpd": Count,
    "hardship_end_date": None,
    "hardship_flag": Discrete,
    "hardship_last_payment_amount": Real,
    "hardship_length": Discrete,
    "hardship_loan_status": Discrete,
    "hardship_payoff_balance_amount": Real,
    "hardship_reason": Discrete,
    "hardship_start_date": None,
    "hardship_status": Discrete,
    "hardship_type": Discrete,
    "home_ownership": Discrete,
    "id": None,
    "il_util": Real,
    "initial_list_status": Discrete,
    "inq_fi": Count,
    "inq_last_12m": Count,
    "inq_last_6mths": Count,
    "installment": Real,
    "int_rate": Real,
    "issue_d": None,
    "last_credit_pull_d": None,
    "last_pymnt_amnt": Real,
    "last_pymnt_d": None,
    "loan_amnt": Real,
    "loan_status": Discrete,
    "max_bal_bc": Real,
    "member_id": None,
    "mo_sin_old_il_acct": Real,
    "mo_sin_old_rev_tl_op": Real,
    "mo_sin_rcnt_rev_tl_op": Real,
    "mo_sin_rcnt_tl": Real,
    "mort_acc": Count,
    "mths_since_last_delinq": Real,
    "mths_since_last_major_derog": Real,
    "mths_since_last_record": Real,
    "mths_since_rcnt_il": Real,
    "mths_since_recent_bc": Real,
    "mths_since_recent_bc_dlq": Real,
    "mths_since_recent_inq": Real,
    "mths_since_recent_revol_delinq": Real,
    "next_pymnt_d": None,
    "num_accts_ever_120_pd": Count,
    "num_actv_bc_tl": Count,
    "num_actv_rev_tl": Count,
    "num_bc_sats": Count,
    "num_bc_tl": Count,
    "num_il_tl": Count,
    "num_op_rev_tl": Count,
    "num_rev_accts": Count,
    "num_rev_tl_bal_gt_0": Count,
    "num_sats": Count,
    "num_tl_120dpd_2m": Count,
    "num_tl_30dpd": Count,
    "num_tl_90g_dpd_24m": Count,
    "num_tl_op_past_12m": Count,
    "open_acc": Count,
    "open_acc_6m": Count,
    "open_act_il": Count,
    "open_il_12m": Count,
    "open_il_24m": Count,
    "open_rv_12m": Count,
    "open_rv_24m": Count,
    "orig_projected_additional_accrued_interest": None,
    "out_prncp": Real,
    "out_prncp_inv": Real,
    "payment_plan_start_date": None,
    "pct_tl_nvr_dlq": Real,
    "percent_bc_gt_75": Real,
    "policy_code": Discrete,
    "pub_rec": Count,
    "pub_rec_bankruptcies": Count,
    "purpose": Discrete,
    "pymnt_plan": Discrete,
    "recoveries": None,
    "revol_bal": Real,
    "revol_bal_joint": None,
    "revol_util": Real,
    "sec_app_chargeoff_within_12_mths": Count,
    "sec_app_collections_12_mths_ex_med": Count,
    "sec_app_earliest_cr_line": None,
    "sec_app_inq_last_6mths": Count,
    "sec_app_mort_acc": Count,
    "sec_app_mths_since_last_major_derog": Count,
    "sec_app_num_rev_accts": Count,
    "sec_app_open_acc": Count,
    "sec_app_open_act_il": Count,
    "sec_app_revol_util": Real,
    "settlement_amount": Real,
    "settlement_date": None,
    "settlement_percentage": Real,
    "settlement_status": Discrete,
    "settlement_term": Count,
    "sub_grade": Discrete,
    "tax_liens": Count,
    "term": Discrete,
    "title": Discrete,
    "tot_coll_amt": Real,
    "tot_cur_bal": Real,
    "tot_hi_cred_lim": Real,
    "total_acc": Count,
    "total_bal_ex_mort": Real,
    "total_bal_il": Real,
    "total_bc_limit": Real,
    "total_cu_tl": Count,
    "total_il_high_credit_limit": Real,
    "total_pymnt": Real,
    "total_pymnt_inv": Real,
    "total_rec_int": Real,
    "total_rec_late_fee": Real,
    "total_rec_prncp": Real,
    "total_rev_hi_lim": Real,
    "url": None,
    "verification_status": Discrete,
    "verification_status_joint": Discrete,
    "zip_code": Discrete,
}


def load_lending(args):
    """
    See https://www.kaggle.com/wendykan/lending-club-loan-data
    """
    # Convert to torch.
    num_rows = min(2260668, args.max_num_rows)
    cache_filename = os.path.join(DATA, "lending.{}.pkl".format(num_rows))
    if os.path.exists(cache_filename):
        dataset = load_object(cache_filename)
    else:
        raw_dir = os.path.join(RAWDATA, "kaggle-lending-club")
        raw_filename = os.path.join(raw_dir, "loan.csv")
        if not os.path.exists(raw_filename):
            mkdir_p(raw_dir)
            raise ValueError("Data missing\n"
                             "Please navigate to \n"
                             "  https://www.kaggle.com/wendykan/lending-club-loan-data\n"
                             "and download loan.csv in {}".format(os.path.abspath(raw_dir)))
        names = [name for name, typ in LENDING_SCHEMA.items() if typ is not None]
        names.sort()
        positions = {name: pos for pos, name in enumerate(names)}
        num_cols = len(names)
        data = torch.zeros(num_rows, num_cols, dtype=torch.float)
        mask = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
        supports = {name: defaultdict(set)
                    for name in names if LENDING_SCHEMA[name] is Discrete}
        with open(raw_filename) as f:
            reader = csv.reader(f)
            header = [name.strip() for name in next(reader)]
            logging.debug(header)
            assert set(names) <= set(header), "invalid schema"
            js = [positions.get(name) for name in header]
            types = [LENDING_SCHEMA.get(name) for name in header]
            supps = [supports.get(name) for name in header]
            cell_count = 0
            for i, row in enumerate(reader):
                if i == num_rows:
                    break
                for name, cell, typ, support, j in zip(header, row, types, supps, js):
                    if typ is None or not cell:
                        continue
                    if typ is Real:
                        value = float(cell)
                    else:
                        value = support.setdefault(cell, len(support))
                    data[i, j] = value
                    mask[i, j] = True
                    cell_count += 1
                if i % max(1, num_rows // 100) == 0:
                    sys.stderr.write(".")
                    sys.stderr.flush()
        logging.info("loaded {} rows x {} features".format(data.size(0), data.size(1)))
        logging.info("observed {} / {} = {:.1f}% cells".format(
            cell_count, data.numel(), 100. * cell_count / data.numel()))
        logging.debug("\n".join(
            ["Cardinalities:"] +
            ["{: >10} {}".format(len(support), name)
             for name, support in sorted(supports.items())]))

        perm = torch.randperm(num_rows)
        data = data[perm].t().contiguous()
        mask = mask[perm].t().contiguous()
        dataset = {
            "names": names,
            "data": data,
            "mask": mask,
            "supports": supports,
            "args": args,
        }
        save_object(dataset, cache_filename)

    # Format columns.
    features = []
    data = []
    mask = []
    for name, col_data, col_mask in zip(dataset["names"], dataset["data"], dataset["mask"]):
        if not col_mask.any():
            logging.debug("Dropping empty feature: {}".format(name))
            continue
        if col_mask.all():
            col_mask = True
        else:
            # Add a presence/absence feature.
            name_nz = "{}_nz".format(name)
            logging.debug("Adding presence feature: {}".format(name_nz))
            feature = Boolean(name_nz)
            features.append(feature)
            data.append(col_mask.to(feature.dtype))
            mask.append(True)

        # Add the feature.
        typ = LENDING_SCHEMA[name]
        if typ is Discrete:
            feature = typ(name, len(dataset["supports"][name]))
        else:
            feature = typ(name)
        features.append(feature)
        data.append(col_data.to(feature.dtype))
        mask.append(col_mask)
    return features, data, mask


# See https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info
COVERTYPE_SCHEMA = OrderedDict([
    ("Elevation", Real),
    ("Aspect", Real),
    ("Slope", Real),
    ("Horizontal_Distance_To_Hydrology", Real),
    ("Vertical_Distance_To_Hydrology", Real),
    ("Horizontal_Distance_To_Roadways", Real),
    ("Hillshade_9am", Real),
    ("Hillshade_Noon", Real),
    ("Hillshade_3pm", Real),
    ("Horizontal_Distance_To_Fire_Points", Real),
])
for i in range(4):
    COVERTYPE_SCHEMA["Wilderness_Area_{}".format(i)] = Boolean
for i in range(40):
    COVERTYPE_SCHEMA["Soil_Type_{}".format(i)] = Boolean
COVERTYPE_SCHEMA["Cover_Type"] = Discrete  # 7 categories
assert len(COVERTYPE_SCHEMA) == 55
del i


def load_covertype(args):
    """
    See https://archive.ics.uci.edu/ml/datasets/Covertype
    """
    # Convert to torch.
    num_rows = min(581012, args.max_num_rows)
    cache_filename = os.path.join(DATA, "covertype.{}.pkl".format(num_rows))
    if os.path.exists(cache_filename):
        dataset = load_object(cache_filename)
    else:
        raw_dir = os.path.join(RAWDATA, "uci-covertype")
        raw_filename = os.path.join(raw_dir, "covtype.data")
        if not os.path.exists(raw_filename):
            logging.info("Downloading covertype dataset")
            mkdir_p(raw_dir)
            zip_filename = os.path.join(raw_dir, "covtype.data.gz")
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
                zip_filename)
            subprocess.check_call(["gunzip", "-k", zip_filename])

        header = list(COVERTYPE_SCHEMA.keys())
        logging.debug(header)
        types = list(COVERTYPE_SCHEMA.values())
        num_cols = len(COVERTYPE_SCHEMA)
        data = [torch.zeros(num_rows, dtype=typ.dtype) for typ in types]
        mask = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
        supports = [defaultdict(set) for typ in types]
        with open(raw_filename) as f:
            reader = csv.reader(f)
            cell_count = 0
            for i, row in enumerate(reader):
                if i == num_rows:
                    break
                for j, (cell, typ, support, col) in enumerate(zip(row, types, supports, data)):
                    if typ is None or not cell:
                        continue
                    if typ is Discrete:
                        value = support.setdefault(cell, len(support))
                    else:
                        value = float(cell)
                    data[j][i] = value
                    mask[i, j] = True
                    cell_count += 1
                if i % max(1, num_rows // 100) == 0:
                    sys.stderr.write(".")
                    sys.stderr.flush()
        logging.info("loaded {} rows x {} features".format(num_rows, num_cols))
        logging.debug("\n".join(
            ["Cardinalities:"] +
            ["{: >10} {}".format(len(support), name)
             for name, support, typ in zip(header, supports, types)
             if typ is Discrete]))

        perm = torch.randperm(num_rows)
        data = [col[perm].contiguous() for col in data]
        mask = mask[perm].t().contiguous()
        dataset = {
            "names": header,
            "data": data,
            "mask": mask,
            "supports": dict(zip(header, supports)),
            "args": args,
        }
        save_object(dataset, cache_filename)

    # Format columns.
    features = []
    data = []
    mask = []
    for name, col_data, col_mask in zip(dataset["names"], dataset["data"], dataset["mask"]):
        if not col_mask.any():
            logging.debug("Dropping empty feature: {}".format(name))
            continue

        # Add the feature.
        typ = COVERTYPE_SCHEMA[name]
        if typ is Discrete:
            feature = typ(name, len(dataset["supports"][name]))
        else:
            feature = typ(name)
        features.append(feature)
        data.append(col_data.to(feature.dtype))
        mask.append(col_mask)
    return features, data, mask


def load_molecules(args):
    """
    A subset of the Harvard Clean Energy Project dataset.
    This is apparently a privately processed subset of the public dataset.
    Ask Tang for details (thang.bui@uber.com).
    """
    # Convert to torch.
    num_rows = min(60000, args.max_num_rows)
    cache_filename = os.path.join(DATA, "molecules.{}.pkl".format(num_rows))
    if os.path.exists(cache_filename):
        dataset = load_object(cache_filename)
    else:
        raw_dir = os.path.join(RAWDATA, "harvard-cep")
        if not os.path.exists(raw_dir):
            raise ValueError("Data missing\n"
                             "Please get raw data from Martin or Fritz")

        grid = np.load(os.path.join(RAWDATA, "harvard-cep", "grid.npy"))
        y_grid = np.load(os.path.join(RAWDATA, "harvard-cep", "y_grid.npy"))
        grid = grid[:args.max_num_rows]
        y_grid = y_grid[:args.max_num_rows]
        grid = torch.tensor(grid, dtype=torch.long)
        y_grid = torch.tensor(y_grid)

        num_rows = len(grid)
        booleans = torch.zeros(num_rows, 512, dtype=torch.float)
        booleans.scatter_add_(1, grid.clamp(min=0), (grid != -1).float())
        booleans = booleans.t().contiguous()
        real = torch.tensor(y_grid)

        names = ["b{}".format(i) for i in range(len(booleans))]
        names.append("real")

        logging.info("loaded {} rows x 513 features".format(num_rows))

        perm = torch.randperm(num_rows)
        data = [col[perm].contiguous() for col in list(booleans) + [real]]
        dataset = {
            "names": names,
            "data": data,
            "args": args,
        }
        save_object(dataset, cache_filename)

    # Format columns.
    features = []
    data = []
    mask = []
    for name, col in zip(dataset["names"], dataset["data"]):
        feature = (Boolean if name.startswith("b") else Real)(name)
        features.append(feature)
        data.append(col.to(feature.dtype))
        mask.append(True)
    return features, data, mask


def partition_data(data, mask, target_size):
    """
    Iterates over minibatches of data, attempting to make each minibatch as
    large as possible up to ``target_size``.

    :param data: Either a single contiguous ``torch.Tensor``s or a
        heterogeneous list of ``torch.Tensors``.
    :param data: Either a single contiguous ``torch.Tensor``s or a
        heterogeneous list of either ``torch.Tensors`` or bools.
    """
    assert len(data) == len(mask)
    assert all(col is not None for col in data)
    num_rows = len(data[0])

    begin = 0
    while begin < num_rows:
        end = begin + target_size

        if isinstance(data, torch.Tensor):
            batch_data = data[:, begin: end]
        else:
            batch_data = [col[begin: end] for col in data]

        if isinstance(mask, torch.Tensor):
            batch_mask = mask[:, begin: end]
        else:
            batch_mask = [col if isinstance(col, bool) else col[begin: end]
                          for col in mask]

        yield batch_data, batch_mask
        begin = end
