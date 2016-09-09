from scipy.sparse import *
from batch_MF import train
from online_updates import update
from util import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    try:
        ratings_dataset = np.load('ml-1m/ratings.npy')
    except:
        ratings_dataset = np.loadtxt("ml-1m/ratings.dat", dtype=np.int32, delimiter='::', usecols=(0, 1, 2))
        np.save('data/ratings', ratings_dataset)

    row = ratings_dataset[:, 0] - 1
    col = ratings_dataset[:, 1] - 1
    data = ratings_dataset[:, 2]
    batch_matrix = coo_matrix((data, (row, col))).toarray()

    N = len(batch_matrix)
    M = len(batch_matrix[0])
    K = 10

    try:
        train_mask = np.load('data/train_mask.npy')
        test_mask = np.load('data/test_mask.npy')

    except:
        train_mask = np.ones((N, M))
        test_mask = np.zeros((N,M))
        users_test = np.random.choice(N, int((N-1)/10), replace=False)
        test_mask[users_test, :] = 1
        train_mask -= test_mask
        np.save('data/train_mask', train_mask)
        np.save('data/test_mask', test_mask)

    u_batch, v_batch = train(batch_matrix, train_mask, N, M, K)

    users = np.unique(np.nonzero(train_mask)[0])
    items = np.unique(np.nonzero(batch_matrix[users, :])[1])

    train_mat = batch_matrix*train_mask

    nz_ratings = non_zero_matrix(train_mat)
    bias = np.sum(train_mat) / np.sum(nz_ratings)

    f = np.dot(u_batch, v_batch.T) + bias
    rmse_train = calc_rmse(train_mat, f)

    updates_matrix = batch_matrix*test_mask

    try:
        test_u = np.load('data/tu_mask.npy')
        val_u = np.load('data/vu_mask.npy')
    except:
        test_u, val_u = build_training_valuation(updates_matrix)
        np.save('data/tu_mask', test_u)
        np.save('data/vu_mask', val_u)
    u_online, v_online = update(batch_matrix, test_u, u_batch, v_batch, bias)

    f = np.dot(u_online, v_online.T) + bias
    # f = np.maximum(np.minimum(f, 5), 1)
    rmse_train_up = calc_rmse(batch_matrix*test_u, f)
    rmse_test_up = calc_rmse(batch_matrix*val_u, f)
    indexes = np.unique(np.nonzero(val_u)[0])
    y = np.zeros(len(indexes))
    for index, i in enumerate(indexes):
        rmse_test = calc_rmse(batch_matrix[i, :]*test_u[i, :], f[i, :])
        y[index]  = rmse_test
    plt.plot(indexes, y, 'ro')
    # plt.axis([5940, 6040, 0.10, 5])
    plt.show()
    plt.savefig('plots/rmse_online_test_scatter.pdf', bbox_inches='tight')
    plt.savefig('plots/rmse_online_test_scatter.png', bbox_inches='tight')
    pass
