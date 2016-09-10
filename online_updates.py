import matplotlib.pyplot as plt
from util import *
import random as rnd


def user_update(u_i, v, bias, profile, epochs=200, alpha0=0.015, beta=0.001):
    profile = np.reshape(profile, (1, len(profile)))
    u_i = np.reshape(u_i, (1, len(u_i)))
    nz_profile = non_zero_matrix(profile)
    vel_u = np.zeros_like(u_i)
    y = np.zeros(epochs)
    for epoch in range(epochs):
        f = (np.dot(u_i, v.T) + bias) * nz_profile
        err = profile - f

        alpha = max(alpha0 / (1 + (epoch / 150)), 0.01)
        mu = min(0.8, 1 / (1 + np.exp(-epoch / 50)))

        u_ahead = u_i + (mu * vel_u)

        delta__u = np.dot(2 * alpha * err, alpha * v) - alpha * beta * u_ahead

        vel_u *= mu
        vel_u += delta__u
        u_i += vel_u

    #     rmse_tot = calc_rmse(profile, f)
    #     y[epoch] = rmse_tot
    # plt.plot(np.arange(epochs), y)
    # plt.show()
    return u_i


def train_incremental_updates(ratings, test_ratings, n_u, n_v, bias):
    unk_user = np.unique(np.nonzero(ratings)[0])
    y = np.zeros(50)
    y_test = np.zeros(50)
    shuffled_profiles = []
    for us in unk_user:
        shuffled_profile = np.random.permutation(np.nonzero(ratings[us, :])[0])
        shuffled_profiles.append(shuffled_profile)
    for m in range(50):
        se = np.zeros(len(unk_user))
        se_len = np.zeros(len(unk_user))
        for index, i in enumerate(unk_user):
            profile_u = shuffled_profiles[index][:m + 1]#np.nonzero(ratings[i, :])[0][:m + 1]
            err_rij = ratings[i, profile_u[-1]] - (np.dot(n_u[i, :], n_v[profile_u[-1], :].T) + bias)
            prob = np.tanh(err_rij ** 2)
            if prob > min(0.89,rnd.random()) and len(profile_u) == m + 1:
                u_i = user_update(n_u[i, :], n_v[profile_u, :], bias, ratings[i, profile_u], epochs=200)
                n_u[i, :] = u_i
            if 45 <= len(shuffled_profiles[index]) <= 50:
                nz_i = non_zero_matrix(test_ratings[i, :])
                err = (test_ratings[i, :] - ((np.dot(n_u[i, :], n_v.T) + bias) * nz_i)) ** 2
                se[index] = np.sum(err)
                se_len[index] = np.sum(nz_i)

        f = (np.dot(n_u[unk_user, :], n_v.T) + bias)
        rmse_tot = calc_rmse(ratings[unk_user, :], f)
        y[m] = rmse_tot
        rmse_test = (np.sum(se) / np.sum(se_len)) ** (1 / 2)
        y_test[m] = rmse_test

    plt.plot(np.arange(50), y_test)
    plt.show()
    plt.plot(np.arange(50), y)
    plt.show()
    # plt.savefig('plots/rmse_online.png', bbox_inches='tight')
    # plt.savefig('plots/rmse_online_test.png', bbox_inches='tight')
    return n_u, n_v


def update(updates_matrix, valuation_matrix, u_batch, v_batch, bias):
    try:
        u_online = np.load('data/u_online.npy')
        v_online = np.load('data/v_online.npy')
    except:
        u_online, v_online = train_incremental_updates(updates_matrix, valuation_matrix, u_batch, v_batch, bias)
        np.save('data/u_online', u_online)
        np.save('data/v_online', v_online)
    return u_online, v_online
