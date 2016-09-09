import matplotlib.pyplot as plt
from util import *


def matrix_factorization(ratings, u, v, epochs= 200, alpha0=0.023, beta=0.001):
    y = np.zeros(epochs)

    nz_ratings = non_zero_matrix(ratings)
    bias = np.sum(ratings) / np.sum(nz_ratings)

    # range_rows = len(ratings)
    # prob_choice_row = np.ones(range_rows) / range_rows
    # n_rows = 800
    # range_cols = len(ratings[0])
    # prob_choice_col = np.ones(range_cols) / range_cols
    # n_cols = 2000

    vel_u = np.zeros_like(u)
    vel_v = np.zeros_like(v)

    for epoch in range(epochs):
        #
        # rows = np.random.choice(range_rows, n_rows, replace=False, p=prob_choice_row)
        # cols = np.random.choice(range_cols, n_cols, replace=False, p=prob_choice_col)

        # err_mat = ((ratings - f)*nz_ratings)**2
        # err_rows = np.sum(err_mat, axis=1)
        # err_r_t = 0.01*np.max(err_rows)+ 0.99*np.min(err_rows)
        # err_cols = np.sum(err_mat, axis=0)
        # err_c_t = 0.008 * np.max(err_cols) + 0.992 * np.min(err_cols)
        # rows = np.where(err_rows>= err_r_t)[0]
        # # cols = np.where(err_cols >= err_c_t)[0]
        # # r = ratings[np.ix_(rows, cols)]
        # nz_r = non_zero_matrix(r)

        alpha = max(alpha0 / (1 + (epoch / 250)), 0.01)
        mu = min(0.99, 1.4 / (1 + np.exp(-epoch / 100)))

        f = (np.dot(u, v.T) + bias) * nz_ratings
        r_m_s_e = calc_rmse(ratings, f)
        y[epoch] = r_m_s_e

        err = ratings - f
        u_ahead = u + (mu * vel_u)
        v_ahead = v + (mu * vel_v)

        delta__u = np.dot(2 * alpha * err, alpha * v_ahead) - alpha * beta * u_ahead
        delta__v = np.dot(2 * alpha * err.T, alpha * u_ahead) - alpha * beta * v_ahead

        vel_u = (mu * vel_u) + delta__u
        vel_v = (mu * vel_v) + delta__v

        u += vel_u
        v += vel_v

        # minus = prob_choice_row[rows] * 0.999
        # sum = np.sum(minus) / (range_rows - n_rows)
        # prob_choice_row[rows] -= minus
        # mask = np.ones_like(prob_choice_row, dtype=bool)
        # mask[rows] = False
        # prob_choice_row[mask] += sum
        #
        # minus = prob_choice_col[cols] * 0.9
        # sum = np.sum(minus) / (range_cols - n_cols)
        # prob_choice_col[cols] -= minus
        # mask = np.ones_like(prob_choice_col, dtype=bool)
        # mask[cols] = False
        # prob_choice_col[mask] += sum

    plt.plot(np.arange(epochs), y)
    plt.show()
    plt.savefig('plots/rmse_batch_test.png', bbox_inches='tight')

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
