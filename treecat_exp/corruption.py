from __future__ import absolute_import, division, print_function

import math

import torch


def random_mask(num_rows, num_cols, prob):
    """
    Generates a random mask with given size and density of ``True``.
    This aims to ensure each row has nearly ``num_cols * prob`` cells
    with value ``True``, at most one more or fewer.
    """
    assert 0 < prob and prob < 1

    # Split into two batches, just more and just less than prob.
    num_cols_0 = int(math.floor(prob * num_cols))
    num_cols_1 = int(math.ceil(prob * num_cols))
    if num_cols_0 == num_cols_1:
        num_rows_0 = num_rows // 2
    else:
        num_rows_0 = int(round(num_rows * (num_cols_1 - prob * num_cols)))
    num_rows_1 = num_rows - num_rows_0
    # Note this assertion may fail for very small datasets.
    assert abs(prob -
               (num_rows_0 * num_cols_0 + num_rows_1 * num_cols_1) /
               (num_rows * num_cols)) < 1e-4

    # Sample from a multinomial.
    result = torch.zeros(num_cols, num_rows, dtype=torch.uint8)
    if num_rows_0:
        uniform = torch.tensor(1.).expand(num_rows_0, num_cols)
        samples = torch.multinomial(uniform, num_cols_0, replacement=False)
        result[samples, torch.arange(num_rows_0).unsqueeze(-1)] = 1
    if num_rows_1:
        uniform = torch.tensor(1.).expand(num_rows_1, num_cols)
        samples = torch.multinomial(uniform, num_cols_1, replacement=False)
        result[samples, torch.arange(num_rows_0, num_rows).unsqueeze(-1)] = 1
    result = result[:, torch.randperm(num_rows)]
    assert result.shape == (num_cols, num_rows)
    assert abs(result.float().mean() - prob) < 1e-4
    return result


def corrupt(data, mask, delete_prob=0, replace_prob=0):
    """
    Corrupts a column-oriented dataset by:

    1.  randomly removing cells with probability ``delete_prob``, then
    2.  replacing random cells with other cells drawn from the marginal, at a
        rate of ``replace_prob``. This produces outliers that can only be
        detected by examining other cells in each row.
    """
    assert 0 <= delete_prob and delete_prob < 1
    assert 0 <= replace_prob and replace_prob < 1
    assert len(data) == len(mask)
    num_cols = len(data)
    num_rows = len(data[0])
    corrupted_data = list(data)
    corrupted_mask = list(mask)

    # Delete random cells.
    if delete_prob == 0:
        delete_mask = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
    else:
        delete_mask = random_mask(num_rows, num_cols, delete_prob)
        corrupted_mask = []
        for i, col in enumerate(mask):
            if isinstance(col, torch.Tensor):
                col = col & ~delete_mask[i]
            else:
                assert col is True
                col = ~delete_mask[i]
            corrupted_mask.append(col)

    # Replace random cells with random other cells.
    # This produces outliers that are difficult to detect.
    if replace_prob == 0:
        replace_mask = torch.zeros(num_rows, num_cols, dtype=torch.uint8)
    else:
        replace_mask = random_mask(num_rows, num_cols, replace_prob)
        replace_mask &= ~delete_mask
        for i, col in enumerate(data):
            random_values = col if mask[i] is True else col[mask[i]]
            random_values = random_values[torch.randperm[len(random_values)]]
            col = col.clone()
            col[replace_mask] = random_values[:replace_mask.long().sum()]
            corrupted_mask[i] = col

    return {
        "data": corrupted_data,
        "mask": corrupted_mask,
        "delete_mask": delete_mask,
        "replace_mask": replace_mask,
    }
