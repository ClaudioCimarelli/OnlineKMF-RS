import matplotlib.pyplot as plt
from util import *


def user_update(u_i, v, bias, profile, epochs=100, alpha=0.019, beta=0.009, mu = 0.7):

    y = np.zeros(epochs)

    profile = np.reshape(profile,(1,len(profile)))
    u_i = np.reshape(u_i, (1, len(u_i)))
    nz_profile = non_zero_matrix(profile)
    vel_u = np.zeros_like(u_i)
    for epoch in range(epochs):
        f = (np.dot(u_i, v.T) + bias) * nz_profile
        err = profile - f

        u_ahead = u_i + (mu * vel_u)

        rmse = calc_rmse(profile, f, nz_profile)
        y[epoch] = rmse

        delta__u = np.dot(2 * alpha * err, alpha * v) - alpha * beta * u_ahead
        # delta__v = (np.dot(2 * alpha * err.T, alpha * np.reshape(u_i[i, :], (1, K))) - alpha * beta * v[:len(profile), :])

        vel_u *= mu
        vel_u += delta__u
        u_i += vel_u
        # v[:len(profile), :] += delta__v


    plt.plot(np.arange(epochs), y)
    plt.show()
    # plt.savefig('data/rmse_online.pdf', bbox_inches='tight')
    # plt.savefig('data/rmse_online.png', bbox_inches='tight')
    return u_i


def train_incremental_updates(ratings, updates, n_u, n_v):
    K = len(n_u[0])
    ratings_nz = non_zero_matrix(ratings)
    bias = np.sum(ratings) / np.sum(ratings_nz)

    new_users = np.unique(np.where(updates > 0)[0])
    for i in new_users:

        if i > len(n_u) - 1:
            n_u = np.append(n_u, np.random.uniform(-0.05,0.05, (i - (len(n_u) - 1), K)), axis=0)
            new_user = np.zeros((i - (len(ratings) - 1), len(ratings[0])))
            ratings = np.append(ratings, new_user, axis=0)
        profile_u = np.where(updates[i, :] > 0)[0]

        for index, j in enumerate(profile_u):
            ratings[i, j] = updates[i, j]
            #if 0.93 ** index > rnd.random():
                #n_u, n_v = user_update(n_u, n_v, bias, ratings[i, profile_u[:index + 1]], i)
        u_i = user_update(n_u[i,:], n_v, bias, ratings[i, :])
        n_u[i, :] = u_i

    return ratings, n_u, n_v


def update(batch_matrix, updates_matrix, u_batch, v_batch):
    try:
        updated_matrix = np.load('data/updated_matrix.npy')
        u_online = np.load('data/u_online.npy')
        v_online = np.load('data/v_online.npy')
    except:
        updated_matrix, u_online, v_online = train_incremental_updates(batch_matrix, updates_matrix, u_batch, v_batch)
        np.save('data/updated_matrix', updated_matrix)
        np.save('data/u_online', u_online)
        np.save('data/v_online', v_online)
    return updated_matrix, u_online, v_online
