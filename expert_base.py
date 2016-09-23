from util import *
from batch_MF import train
from cluster_build import build_clusters
from user_based_predictions import user_based_pred


if __name__ == "__main__":

    batch_matrix = load_data()
    nz_batch = non_zero_matrix(batch_matrix)
    ratings_count = np.sum(nz_batch, axis=1)
    sort_by_ratings_index = np.argsort(ratings_count, kind='heapsort')
    experts_index = np.sort(sort_by_ratings_index[-4000:])
    private_users_index = np.sort(sort_by_ratings_index[:-4000])
    private_users_matrix = batch_matrix[private_users_index, :]

    mask = np.zeros_like(batch_matrix)
    mask[experts_index, :] = 1

    N = len(batch_matrix)
    M = len(batch_matrix[0])
    K = 40
    u_batch, v_batch = train(batch_matrix*mask, N, M, K, suffix_name='experts')

    clusters, clusters_index = build_clusters(private_users_matrix)

    rmse = np.zeros(11)

    for i, cluster in enumerate(clusters):
        train_mask, test_mask = build_test_train_masks(cluster)
        pred = user_based_pred(cluster*train_mask)
        rmse[i] = calc_rmse(cluster*test_mask, pred)

    train_mask, test_mask = build_test_train_masks(batch_matrix)
    pred = user_based_pred(batch_matrix * train_mask)
    rmse[10] = calc_rmse(batch_matrix * test_mask, pred)
    pass


