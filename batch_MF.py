import matplotlib.pyplot as plt
from util import *


def matrix_factorization(ratings, u, v, epochs=300, alpha=0.015, beta=0.009, mu=0.7, T=50):

    y = np.zeros(epochs)

    nz_ratings = non_zero_matrix(ratings)
    bias = np.sum(ratings) / np.sum(nz_ratings)

    range_rows = np.arange(len(ratings))
    prob_choice_row = np.ones_like(range_rows) / len(range_rows)
    range_cols = np.arange(len(ratings[0]))
    prob_choice_col = np.ones_like(range_cols) / len(range_cols)

    for epoch in range(epochs):

        vel_u = np.zeros_like(u)
        vel_v = np.zeros_like(v)

        rows = np.random.choice(range_rows, 400, replace=False, p=prob_choice_row)
        cols = np.random.choice(range_cols, 1000, replace=False, p=prob_choice_col)
        r = ratings[np.ix_(rows, cols)]
        nz_r = non_zero_matrix(r)
        # count = np.count_nonzero(r)
        for step in range(T):
            f = (np.dot(u[rows, :], v[cols, :].T) + bias) * nz_r

            err = r - f
            u_ahead = u[rows, :] + (mu * vel_u[rows, :])
            v_ahead = v[cols, :] + (mu * vel_v[cols, :])

            delta__u = np.dot(2 * alpha * err, alpha * v_ahead) - alpha * beta * u_ahead
            delta__v = np.dot(2 * alpha * err.T, alpha * u_ahead) - alpha * beta * v_ahead

            vel_u[rows, :] = (mu * vel_u[rows, :]) + delta__u
            vel_v[cols, :] = (mu * vel_v[cols, :]) + delta__v

            u[rows, :] += vel_u[rows, :]
            v[cols, :] += vel_v[cols, :]

        varz = prob_choice_row[rows] * 0.6
        sum = np.sum(varz) / (len(range_rows) - len(rows))
        prob_choice_row[rows] -= varz
        mask = np.ones_like(prob_choice_row, dtype=bool)
        mask[rows] = False
        prob_choice_row[mask] += sum

        varz = prob_choice_col[cols] * 0.6
        sum = np.sum(varz) / (len(range_cols) - len(cols))
        prob_choice_col[cols] -= varz
        mask = np.ones_like(prob_choice_col, dtype=bool)
        mask[cols] = False
        prob_choice_col[mask] += sum

        # r_m_s_e = calc_rmse(r, f, non_zero_matrix(r))
        f = (np.dot(u, v.T) + bias)
        r_m_s_e = calc_rmse(ratings, f, nz_ratings)
        y[epoch] = r_m_s_e

    f = (np.dot(u, v.T) + bias)
    r_m_s_e = calc_rmse(ratings, f, nz_ratings)

    plt.plot(np.arange(epochs)[-150:], y[-150:])
    plt.show()
    plt.savefig('data/rmse_batch_test.pdf', bbox_inches='tight')
    plt.savefig('data/rmse_batch_test.png', bbox_inches='tight')

    return u, v


def train(ratings, N, M, K):
    try:
        u_b = np.load('data/u_batch.npy')
        v_b = np.load('data/v_batch.npy')
    except:
        u_b = np.random.uniform(-0.05, 0.05, (N, K))
        v_b = np.random.uniform(-0.05, 0.05, (M, K))
        u_b, v_b = matrix_factorization(ratings, u_b, v_b)
        np.save('data/u_batch', u_b)
        np.save('data/v_batch', v_b)

    return u_b, v_b
