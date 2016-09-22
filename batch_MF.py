import matplotlib.pyplot as plt
from util import *


def matrix_factorization(ratings, u, v, epochs= 300, alpha0=0.023, beta=0.07):
    y = np.zeros(epochs)

    nz_ratings = non_zero_matrix(ratings)
    bias = np.sum(ratings) / np.sum(nz_ratings)

    vel_u = np.zeros_like(u)
    vel_v = np.zeros_like(v)

    u_prev = np.zeros_like(u)
    u_prev[...] = u
    v_prev = np.zeros_like(v)
    v_prev[...] = v

    f = (np.dot(u, v.T) + bias) * nz_ratings
    err = ratings - f

    for epoch in range(epochs):

        alpha = max(alpha0 / (1 + (epoch / 150)), 0.01)
        mu = min(0.999, 1.4 / (1 + np.exp(-epoch / 80)))

        u_ahead = u + (mu * vel_u)
        v_ahead = v + (mu * vel_v)

        delta__u = np.dot(2 * alpha * err, alpha * v_ahead) - (alpha * beta * u_ahead)
        delta__v = np.dot(2 * alpha * err.T, alpha * u_ahead) - (alpha * beta * v_ahead)

        vel_u = (mu * vel_u) + delta__u
        vel_v = (mu * vel_v) + delta__v

        u += vel_u
        v += vel_v

        f = (np.dot(u, v.T) + bias) * nz_ratings
        r_m_s_e = calc_rmse(ratings, f)
        y[epoch] = r_m_s_e
        err = ratings - f

        if epoch >0 and (y[epoch]>y[epoch -1]):
            u[...] = u_prev
            v[...] = v_prev
            vel_u[...] = np.zeros_like(u)
            vel_v[...] = np.zeros_like(v)
        else:
            u_prev[...] = u
            v_prev[...] = v

    plt.plot(np.arange(epochs), y)
    plt.savefig('plots/rmse_batch_test_v4.png', bbox_inches='tight')
    plt.show()

    return u, v


def train(ratings, mask, N, M, K):
    try:
        u_b = np.load('data/u_batch.npy')
        v_b = np.load('data/v_batch.npy')
    except:
        u_b = np.random.uniform(-0.05, 0.05, (N, K))
        v_b = np.random.uniform(-0.05, 0.05, (M, K))
        users = np.unique(np.nonzero(mask)[0])
        items = np.unique(np.nonzero(ratings[users, :])[1])
        u_b[users, :], v_b[items, :] = matrix_factorization(ratings[np.ix_(users, items)], u_b[users, :], v_b[items, :])
        np.save('data/u_batch', u_b)
        np.save('data/v_batch', v_b)

    return u_b, v_b
