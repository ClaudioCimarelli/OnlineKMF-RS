import numpy as np
from scipy.sparse import *


def load_data():
    try:
        ratings_dataset = np.load('ml-1m/ratings.npy')
    except:
        ratings_dataset = np.loadtxt("ml-1m/ratings.dat", dtype=np.int32, delimiter='::', usecols=(0, 1, 2))
        np.save('ml-1m/ratings', ratings_dataset)

    row = ratings_dataset[:, 0] - 1
    col = ratings_dataset[:, 1] - 1
    data = ratings_dataset[:, 2]
    batch_matrix = coo_matrix((data, (row, col))).toarray()
    return batch_matrix


def expert_base(batch_matrix, max_users=4000):
    try:
        experts_index = np.load('data/experts_index.npy')
        private_users_index = np.load('data/private_users_index.npy')
    except:
        nz_batch = non_zero_matrix(batch_matrix)
        ratings_count = np.sum(nz_batch, axis=1)
        sort_by_ratings_index = np.argsort(ratings_count, kind='mergesort')
        experts_index = np.sort(sort_by_ratings_index[-max_users:]).astype('int32')
        private_users_index = np.sort(sort_by_ratings_index[:-max_users]).astype('int32')
        np.save('data/experts_index', experts_index)
        np.save('data/private_users_index', private_users_index)
    return experts_index, private_users_index


def non_zero_matrix(r):
    nz_r = np.zeros_like(r, dtype='float32')
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(r, r, out=nz_r)
        nz_r[...] = np.nan_to_num(nz_r)
    return nz_r


def calc_rmse(real_values, prediction):
    mask = non_zero_matrix(real_values)
    error = (real_values - prediction) * mask
    error **= 2
    RMSE = (np.sum(error) / np.sum(mask)) ** (1 / 2)
    return RMSE


def build_test_train_masks(data_matrix, user_num=None):
    train_mask = np.zeros_like(data_matrix)
    test_mask = np.zeros_like(data_matrix)
    new_users = np.unique(np.nonzero(data_matrix)[0])
    if user_num is not None and user_num < len(new_users):
        np.random.shuffle(new_users)
        new_users = np.array_split(new_users, [user_num])[0]
    for i in new_users:
        items = np.random.permutation(np.nonzero(data_matrix[i, :])[0])
        split = np.array_split(items, [int(len(items) * 8 / 10)])
        train_mask[i, split[0]] = 1
        test_mask[i, split[1]] = 1

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