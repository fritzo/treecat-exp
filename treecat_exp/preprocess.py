from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from collections import defaultdict

import torch
from observations import boston_housing
from pyro.contrib.tabular import Boolean, Discrete, Real
from six.moves import cPickle as pickle

from treecat_exp.util import DATA, RAWDATA

Count = Discrete  # We currently don't handle count data.
Text = None  # We currently don't handle text data.


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
    mask = True
    return dataset["feature"], dataset["data"], mask


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
    mask = True
    for j, support in enumerate(dataset["supports"]):
        if len(support) >= 2:
            name = dataset["header"][j]
            features.append(Discrete(name, len(support)))
            data.append(dataset["data"][:, j].long().contiguous())
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
    mask = True
    for name, col in zip(dataset["names"], dataset["data"]):
        features.append(NEWS_SCHEMA[name](name))
        data.append(col)
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
    num_rows = min(2260668, args.max_num_rows)
    filename = os.path.join(DATA, "lending.{}.pkl".format(num_rows))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        names = [name for name, typ in LENDING_SCHEMA.items() if typ is not None]
        names.sort()
        positions = {name: pos for pos, name in enumerate(names)}
        num_cols = len(names)
        data = torch.zeros(num_rows, num_cols, dtype=torch.float)
        mask = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
        supports = {name: defaultdict(set)
                    for name in names if LENDING_SCHEMA[name] is Discrete}
        with open(os.path.join(RAWDATA, "kaggle-lending-club", "loan.csv")) as f:
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
                if i % (num_rows // 100) == 0:
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
        with open(filename, "wb") as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
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


def partition_data(data, mask, target_size):
    """
    Iterates over minibatches of data, attempting to make each minibatch as
    large as possible up to ``target_size``.
    """
    if mask is True:
        mask = [True] * len(data)
    num_rows = next(col.size(0) for col in data if col is not None)

    begin = 0
    while begin < num_rows:
        end = begin + target_size
        batch_data, batch_mask = [], []

        for col_data, col_mask in zip(data, mask):
            if isinstance(col_mask, torch.Tensor):
                col_mask = col_mask[begin: end]
                if col_mask.any():
                    batch_data.append(col_data[begin: end])
                    batch_mask.append(col_mask)
                    continue
                else:
                    col_mask = False
            batch_data.append(col_data[begin: end] if col_mask else None)
            batch_mask.append(col_mask)

        yield batch_data, batch_mask
        begin = end
