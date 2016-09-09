import matplotlib.pyplot as plt
from util import *
import random as rnd


def user_update(u_i, v, bias, profile, epochs=200, alpha0=0.019, beta=0.001):

    profile = np.reshape(profile,(1,len(profile)))
    u_i = np.reshape(u_i, (1, len(u_i)))
    nz_profile = non_zero_matrix(profile)
    vel_u = np.zeros_like(u_i)
    for epoch in range(epochs):
        f = (np.dot(u_i, v.T) + bias) * nz_profile
        err = profile - f

        alpha = max(alpha0 / (1 + (epoch / 250)), 0.01)
        mu = min(0.9, 1.4 / (1 + np.exp(-epoch / 100)))

        u_ahead = u_i + (mu * vel_u)

        cols = np.nonzero(profile)[1]
        delta__u = np.dot(2 * alpha * err[0, cols], alpha * v[:,:]) - alpha * beta * u_ahead

        vel_u *= mu
        vel_u += delta__u
        u_i += vel_u

    return u_i


def train_incremental_updates(ratings, n_u, n_v, bias):
    unk_user = np.unique(np.nonzero(ratings)[0])
    y = np.zeros(50)
    for m in range(50):
        for i in unk_user:
            profile_u = np.nonzero(ratings[i, :])[0][:m+1]
            if 0.93 ** m > rnd.random():
                u_i = user_update(n_u[i,:], n_v[profile_u,:], bias, ratings[i, profile_u], epochs=10+5*m)
                n_u[i, :] = u_i

        f = (np.dot(n_u[unk_user,:], n_v.T) + bias)
        rmse_tot = calc_rmse(ratings[unk_user, :], f)
        y[m] = rmse_tot
    plt.plot(np.arange(50), y)
    plt.show()
    plt.savefig('plots/rmse_online.pdf', bbox_inches='tight')
    plt.savefig('plots/rmse_online.png', bbox_inches='tight')

    return n_u, n_v


def update(ratings_matrix, updates_mask, u_batch, v_batch, bias, m=50):
    try:
        u_online = np.load('data/u_online.npy')
        v_online = np.load('data/v_online.npy')
    except:
        updates_matrix = ratings_matrix*updates_mask
        u_online, v_online = train_incremental_updates(updates_matrix, u_batch, v_batch, bias)
        np.save('data/u_online', u_online)
        np.save('data/v_online', v_online)
    return u_online, v_online
