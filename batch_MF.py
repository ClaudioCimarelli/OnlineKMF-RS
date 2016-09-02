import matplotlib.pyplot as plt
from util import *


def matrix_factorization(ratings, u, v, epochs=1000, alpha=0.039, beta=0.0):
    y = np.zeros(epochs)
    r = ratings[:500,:2000]
    nz_r = non_zero_matrix(r)
    bias = np.sum(r) / np.sum(nz_r)
    for epoch in range(epochs):
        f = (np.dot(u[:len(r),:], v[:len(r[0]),:].T) + bias)*nz_r
        #f = np.maximum((np.minimum(f,5),1))*nz_r

        err = r - f

        r_m_s_e = calc_rmse(r, f, nz_r)
        y[epoch] = r_m_s_e

        delta__u = np.dot(2 * alpha * err, alpha * v[:len(r[0]),:]) - alpha * beta * u[:len(r),:]
        delta__v = np.dot(2 * alpha * err.T, alpha * u[:len(r),:]) - alpha * beta * v[:len(r[0]),:]

        #alpha *= 0.99999
        u[:len(r),:] += delta__u
        v[:len(r[0]),:] += delta__v

    plt.plot(np.arange(epochs), y)
    plt.show()
    plt.savefig('data/rmse_batch.pdf', bbox_inches='tight')
    plt.savefig('data/rmse_batch.png', bbox_inches='tight')

    return u, v


def train(ratings, N, M, K):
    try:
        u_b = np.load('data/u_batch.npy')
        v_b = np.load('data/v_batch.npy')
    except:
        u_b = np.random.uniform(-0.05,0.05,(N, K))
        v_b = np.random.uniform(-0.05,0.05,(M, K))
        u_b, v_b = matrix_factorization(ratings, u_b, v_b)
        np.save('data/u_batch', u_b)
        np.save('data/v_batch', v_b)

    return u_b, v_b
