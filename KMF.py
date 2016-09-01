import random as rnd
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt


def matrix_factorization(ratings, u, v, steps=500, alpha=0.01, beta=0.002):
    y = np.zeros(steps)
    with np.errstate(divide='ignore', invalid='ignore'):
        nzR = np.nan_to_num(np.divide(ratings, ratings))
    bias = np.sum(ratings) / np.sum(nzR)
    for step in range(steps):
        f_ratings = np.dot(u, v.T) + bias

        err = ratings - (f_ratings * nzR)

        RMSE = calc_rmse(ratings, f_ratings * nzR, nzR)
        y[step] = RMSE

        delta_U = (np.dot(2 * alpha * err, alpha * v) - alpha * beta * u)
        delta_V = (np.dot(2 * alpha * err.T, alpha * u) - alpha * beta * v)

        alpha *= 0.999
        u += delta_U
        v += delta_V

    plt.plot(np.arange(steps), y)
    plt.show()
    return u, v


def train(R, N, M, K):
    try:
        n_u = np.load('nU.npy')
        n_v = np.load('nV.npy')
    except:
        n_u = np.random.rand(N, K)
        n_v = np.random.rand(M, K)
        n_u, n_v = matrix_factorization(R, n_u, n_v)
        np.save('nU', n_u)
        np.save('nV', n_v)

    return n_u, n_v


def user_update(U, V, bias, profile, i, steps=30, alpha=0.01, beta=0.02):
    K = len(U[0])
    for step in range(steps):
        nR = np.dot(U, V.T) + bias
        err = np.reshape(profile,(1,len(profile))) - nR[i, :len(profile)]

        delta__u = (np.dot(2 * alpha * err, alpha * V[:len(profile), :]) - alpha * beta * U[i, :])
        delta__v = (np.dot(2 * alpha * err.T, alpha * np.reshape(U[i, :], (1, K))) - alpha * beta * V[:len(profile), :])

        U[i, :] = delta__u + U[i,:]
        V[:len(profile), :] += delta__v

        # for j in range(len(profile)):
        #     eij = profile[j] - np.dot(u_online[i, :], v_online[:, j])
        #     for k in range(K):
        #         u_online[i, k] += alpha * (2 * eij * v_online[k, j] - beta * u_online[i, k])
        #         v_online[k, j] += alpha * (2 * eij * u_online[i, k] - beta * v_online[k, j])

        alpha *= 0.99
    return U, V


def train_incremental_updates(ratings, updates, n_u, n_v):
    K = len(n_u[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratings_nz = np.nan_to_num(np.divide(ratings, ratings))
    bias = np.sum(ratings) / np.sum(ratings_nz)
    new_users = np.unique(np.where(updates > 0)[0])
    for i in new_users:

        if i > len(n_u) - 1:
            n_u = np.append(n_u, np.random.rand(i - (len(n_u) - 1), K), axis=0)
            new_user = np.zeros((i - (len(ratings) - 1), len(ratings[0])))
            ratings = np.append(ratings, new_user, axis=0)
        profile_u = np.where(updates[i, :] > 0)[0]

        for index, j in enumerate(profile_u):
            ratings[i, j] = updates[i, j]
            if 0.96 ** index > rnd.random():
                n_u, n_v = user_update(n_u, n_v, bias, ratings[i, profile_u[:index + 1]], i)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratings_nz = np.nan_to_num(np.divide(ratings, ratings))
        f_ratings = np.dot(n_u, n_v.T) + bias
        RMSE = calc_rmse(ratings, f_ratings * ratings_nz, ratings_nz)
        print(RMSE)

    return ratings, n_u, n_v


def calc_rmse(real_values, prediction, mask):
    error = (real_values - prediction) * mask
    error **= 2
    RMSE = (np.sum(error) / np.sum(mask)) ** 1 / 2
    return RMSE


def build_updates_masks(updates, user_num):
    try:
        train_mask = np.load('train_mask.npy')
        test_mask = np.load('test_mask.npy')
    except:
        train_mask = np.zeros_like(updates)
        test_mask = np.zeros_like(updates)
        new_users = np.random.permutation(np.unique(np.where(updates > 0)[0]))
        new_users = np.array_split(new_users, [user_num])[0]
        for i in new_users:
            items = np.random.permutation(np.where(updates[i, :] > 0)[0])
            split = np.array_split(items, [int(len(items) * 8 / 10)])
            for j in split[0]:
                train_mask[i, j] = 1
            for j in split[1]:
                test_mask[i, j] = 1
        np.save('train_mask', train_mask)
        np.save('test_mask', test_mask)

    return train_mask, test_mask


if __name__ == "__main__":

    try:
        myarray = np.load('ratings.npy')
    except:
        myarray = np.loadtxt("ml-1m/ratings.dat", dtype=np.int32, delimiter='::', usecols=(0, 1, 2))
        np.save('ratings', myarray)

    train_index = np.where(myarray[:, 0] > 3020)[0].min()
    row = myarray[:train_index, 0] - 1
    col = myarray[:train_index, 1] - 1
    data = myarray[:train_index, 2]
    batch_matrix = coo_matrix((data, (row, col))).toarray()

    N = len(batch_matrix)
    M = len(batch_matrix[0])
    K = 40

    nU, nV = train(batch_matrix, N, M, K)

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     nzR = np.nan_to_num(np.divide(R, R))
    #
    # nR = np.dot(nU, nV.T)
    # RMSE = calc_RMSE(R, nR, nzR)

    row = myarray[train_index:, 0] - 1
    col = myarray[train_index:, 1] - 1
    data = myarray[train_index:, 2]
    updates_matrix = coo_matrix((data, (row, col))).toarray()

    train_mask, test_mask = build_updates_masks(updates_matrix, 100)

    try:
        updated_matrix = np.load('R.npy')
        U = np.load('u_online.npy')
        V = np.load('v_online.npy')
    except:
        updated_matrix, U, V = train_incremental_updates(batch_matrix, updates_matrix * train_mask, nU, nV)
        np.save('R', batch_matrix)
        np.save('u_online', U)
        np.save('v_online', V)
    RMSE = calc_rmse(batch_matrix, np.dot(U, V.T), train_mask[:len(batch_matrix), :])
    print(RMSE)
