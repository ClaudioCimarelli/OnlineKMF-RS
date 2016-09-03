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

    train_index = np.where(ratings_dataset[:, 0] > 5940)[0][0]
    row = ratings_dataset[:train_index, 0] - 1
    col = ratings_dataset[:train_index, 1] - 1
    data = ratings_dataset[:train_index, 2]
    batch_matrix = coo_matrix((data, (row, col))).toarray()

    N = len(batch_matrix)
    M = len(batch_matrix[0])
    K = 40
    try:
        train_mask = np.load('data/train_mask.npy')
        test_mask = np.load('data/test_mask.npy')

    except:
        train_mask, test_mask = build_updates_masks(batch_matrix, user_num=len(batch_matrix))
        np.save('data/train_mask', train_mask)
        np.save('data/test_mask', test_mask)

    u_batch, v_batch = train(batch_matrix * train_mask, N, M, K)

    nz_r = non_zero_matrix(batch_matrix * train_mask)
    bias = np.sum(batch_matrix * train_mask) / np.sum(nz_r)
    f = np.dot(u_batch, v_batch.T) + bias
    rmse_batch_test = calc_rmse(batch_matrix, f, train_mask)

    row = ratings_dataset[train_index:, 0] - 1
    col = ratings_dataset[train_index:, 1] - 1
    data = ratings_dataset[train_index:, 2]
    updates_matrix = coo_matrix((data, (row, col))).toarray()

    try:
        train_mask_updates = np.load('data/train_mask_updates.npy')
        test_mask_updates = np.load('data/test_mask_updates.npy')

    except:
        train_mask, test_mask = build_updates_masks(updates_matrix, len(updates_matrix))
        np.save('data/train_mask_updates', train_mask)
        np.save('data/test_mask_updates', test_mask)

    updated_matrix, u_online, v_online = update(batch_matrix * train_mask, updates_matrix * train_mask_updates, u_batch,
                                                v_batch)

    f = np.dot(u_online, v_online.T) + bias
    updated_matrix +=  (updates_matrix*test_mask_updates)
    f = np.maximum(np.minimum(f, 5), 1)
    rmse_train = calc_rmse(updated_matrix, f, train_mask_updates)
    rmse_test = calc_rmse(updated_matrix, f, test_mask_updates)
    indexes = np.nonzero(test_mask_updates)[0]
    y = np.zeros_like(indexes, dtype=np.float64)
    for index, i in enumerate(indexes):
        rmse_test = calc_rmse(updated_matrix[i, :], f[i, :], test_mask_updates[i, :])
        y[index]  = rmse_test
    plt.plot(indexes, y, 'ro')
    plt.axis([5940, 6040, 0.10, 1.5])
    plt.show()
    pass
