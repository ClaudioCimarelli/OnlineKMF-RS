import matplotlib.pyplot as plt
from util import *


def matrix_factorization(ratings, u, v, epochs=150, alpha0=0.023, beta=0.045):
    nz_ratings = non_zero_matrix(ratings)
    bias = np.divide(np.sum(ratings), np.sum(nz_ratings))

    vel_u = np.zeros_like(u)
    vel_v = np.zeros_like(v)
    f = (np.dot(u, v.T) + bias) * nz_ratings
    err = ratings - f

    for epoch in range(epochs):

        alpha = max(alpha0 / (1 + (epoch / 200)), 0.01)
        mu = min(0.99, 1.2 / (1 + np.exp(-epoch / 100)))

        u_ahead = u + (mu * vel_u)
        v_ahead = v + (mu * vel_v)

        delta__u = np.dot(2 * alpha * err, alpha * v_ahead) - (2 * alpha * beta * u_ahead)
        delta__v = np.dot(2 * alpha * err.T, alpha * u_ahead) - (2 * alpha * beta * v_ahead)

        vel_u[...] = (mu * vel_u) + delta__u
        vel_v[...] = (mu * vel_v) + delta__v

        u[...] += vel_u
        v[...] += vel_v

        f = (np.dot(u, v.T) + bias) * nz_ratings
        err = ratings - f

    return u, v


def train(ratings, N, M, K, suffix_name='batch'):
    try:
        u_b = np.load('data/u_' + suffix_name + '.npy')
        v_b = np.load('data/v_' + suffix_name + '.npy')
    except:
        u_b = np.random.uniform(-0.05, 0.05, (N, K))
        v_b = np.random.uniform(-0.05, 0.05, (M, K))
        users = np.unique(np.nonzero(ratings)[0]).astype('int32')
        items = np.unique(np.nonzero(ratings[users, :])[1]).astype('int32')
        u_b[users, :], v_b[items, :] = matrix_factorization(ratings[np.ix_(users, items)], u_b[users, :], v_b[items, :])
        np.save('data/u_' + suffix_name, u_b)
        np.save('data/v_' + suffix_name, v_b)

    return u_b, v_b
