import numpy as np


def non_zero_matrix(r):
    with np.errstate(divide='ignore', invalid='ignore'):
        nz_r = np.nan_to_num(np.divide(r, r))
    return nz_r


def calc_rmse(real_values, prediction, mask):
    error = (real_values - prediction) * mask
    error **= 2
    RMSE = (np.sum(error) / np.sum(mask)) ** (1 / 2)
    return RMSE


def build_updates_masks(updates, user_num=100):

    train_mask = np.zeros_like(updates)
    test_mask = np.zeros_like(updates)
    new_users = np.unique(np.nonzero(updates)[0])
    if user_num < len(new_users):
        np.random.shuffle(new_users)
        new_users = np.array_split(new_users, [user_num])[0]
    for i in new_users:
        items = np.random.permutation(np.nonzero(updates[i, :])[0])
        split = np.array_split(items, [int(len(items) * 8 / 10)])
        for j in split[0]:
            train_mask[i, j] = 1
        for j in split[1]:
            test_mask[i, j] = 1

    return train_mask, test_mask
