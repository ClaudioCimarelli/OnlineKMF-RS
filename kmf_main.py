from scipy.sparse import *
from batch_MF import train
from online_updates import update
from util import *

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

    u_batch, v_batch = train(batch_matrix, N, M, K)
    nz_r = non_zero_matrix(batch_matrix)
    bias = np.sum(batch_matrix) / np.sum(nz_r)
    f = np.dot(u_batch, v_batch.T)+bias
    rmse_batch = calc_rmse(batch_matrix, f, non_zero_matrix(batch_matrix))

    row = ratings_dataset[train_index:, 0] - 1
    col = ratings_dataset[train_index:, 1] - 1
    data = ratings_dataset[train_index:, 2]
    updates_matrix = coo_matrix((data, (row, col))).toarray()

    train_mask, test_mask = build_updates_masks(updates_matrix)

    updated_matrix, u_online, v_online = update(batch_matrix, updates_matrix, train_mask, u_batch, v_batch)
    f = np.dot(u_online, v_online.T)+bias
    f = np.maximum(np.minimum(f, 5), 1)

    rmse_train = calc_rmse(updated_matrix, f, train_mask[:len(updated_matrix), :])
    rmse_test = calc_rmse(updated_matrix, f, test_mask[:len(updated_matrix), :])
    pass
