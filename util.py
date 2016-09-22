import numpy as np
from scipy.sparse import *


def load_data():
    try:
        ratings_dataset = np.load('ml-1m/ratings.npy')
    except:
        ratings_dataset = np.loadtxt("ml-1m/ratings.dat", dtype=np.int32, delimiter='::', usecols=(0, 1, 2))
        np.save('data/ratings', ratings_dataset)

    row = ratings_dataset[:, 0] - 1
    col = ratings_dataset[:, 1] - 1
    data = ratings_dataset[:, 2]
    batch_matrix = coo_matrix((data, (row, col))).toarray()
    return batch_matrix


def non_zero_matrix(r):
    with np.errstate(divide='ignore', invalid='ignore'):
        nz_r = np.nan_to_num(np.divide(r, r))
    return nz_r


def calc_rmse(real_values, prediction):
    mask = non_zero_matrix(real_values)
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


def build_training_valuation(unknown_users_ratings , m=50):
    new_users = np.unique(np.nonzero(unknown_users_ratings)[0])
    t_u = np.zeros_like(unknown_users_ratings)
    v_u = np.zeros_like(unknown_users_ratings)
    for i in new_users:
        items = np.random.permutation(np.nonzero(unknown_users_ratings[i, :])[0])
        split_index = min(m, len(items)/2)
        split = np.array_split(items, [int(split_index)])
        t_u[i, split[0]] = 1
        v_u[i, split[1]] = 1

    return t_u, v_u