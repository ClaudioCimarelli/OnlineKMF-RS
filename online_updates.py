import random as rnd
import matplotlib.pyplot as plt
from util import *


def user_update(u, v, bias, profile, i, epochs=150, alpha=0.023, beta=0.02):

    y = np.zeros(epochs)
    profile = np.reshape(profile,(1,len(profile)))
    nz_profile = non_zero_matrix(profile)
    for epoch in range(epochs):
        f = (np.dot(u, v.T) + bias)*nz_profile
        err = profile - f[i, :len(profile[0])]

        rmse = calc_rmse(profile, f[i, :len(profile[0])], nz_profile)
        y[epoch] = rmse

        delta__u = np.dot(2 * alpha * err, alpha * v[:len(profile[0]), :]) - alpha * beta * u[i, :]
        #delta__v = (np.dot(2 * alpha * err.T, alpha * np.reshape(u[i, :], (1, K))) - alpha * beta * v[:len(profile), :])

        u[i, :] = delta__u + u[i, :]

        #v[:len(profile), :] += delta__v

        # for j in range(len(profile)):
        #     eij = profile[j] - np.dot(u_online[i, :], v_online[:, j])
        #     for k in range(K):
        #         u_online[i, k] += alpha * (2 * eij * v_online[k, j] - beta * u_online[i, k])
        #         v_online[k, j] += alpha * (2 * eij * u_online[i, k] - beta * v_online[k, j])

    plt.plot(np.arange(epochs), y)
    plt.show()
    plt.savefig('data/rmse_online.pdf', bbox_inches='tight')
    plt.savefig('data/rmse_online.png', bbox_inches='tight')
    return u, v


def train_incremental_updates(ratings, updates, n_u, n_v):
    K = len(n_u[0])
    ratings_nz = non_zero_matrix(ratings)
    bias = np.sum(ratings) / np.sum(ratings_nz)

    new_users = np.unique(np.where(updates > 0)[0])
    for i in new_users:

        if i > len(n_u) - 1:
            n_u = np.append(n_u, np.random.uniform(-0.1,0.1, (i - (len(n_u) - 1), K)), axis=0)
            new_user = np.zeros((i - (len(ratings) - 1), len(ratings[0])))
            ratings = np.append(ratings, new_user, axis=0)
        profile_u = np.where(updates[i, :] > 0)[0]

        for index, j in enumerate(profile_u):
            ratings[i, j] = updates[i, j]
            #if 0.93 ** index > rnd.random():
                #n_u, n_v = user_update(n_u, n_v, bias, ratings[i, profile_u[:index + 1]], i)
        n_u, n_v = user_update(n_u, n_v, bias, ratings[i, :], i)

    return ratings, n_u, n_v


def update(batch_matrix, updates_matrix, train_mask, u_batch, v_batch):
    try:
        updated_matrix = np.load('data/updated_matrix.npy')
        u_online = np.load('data/u_online.npy')
        v_online = np.load('data/v_online.npy')
    except:
        updated_matrix, u_online, v_online = train_incremental_updates(batch_matrix, updates_matrix * train_mask, u_batch, v_batch)
        np.save('data/updated_matrix', updated_matrix)
        np.save('data/u_online', u_online)
        np.save('data/v_online', v_online)
    return updated_matrix, u_online, v_online
