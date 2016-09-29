from util import *
from batch_MF import train
from online_updates import user_update
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
    nz_ratings = non_zero_matrix(batch_matrix*mask)
    bias = np.sum(batch_matrix*mask) / np.sum(nz_ratings)

    clusters, clusters_index = build_clusters(private_users_matrix)

    rmse_user_based = np.zeros(10)
    rmse_imf = np.zeros(10)
    len_clusters = np.zeros(10)
    cluster_dens = np.zeros(10)

    for index, cluster in enumerate(clusters):
        ###train and test set per each cluster
        train_mask, test_mask = build_test_train_masks(cluster)
        ###User based predictions
        pred = user_based_pred(cluster*train_mask)
        pred = np.maximum(np.minimum(pred, 5), 1)
        rmse_user_based[index] = calc_rmse(cluster * test_mask, pred)
        ###IMF predictions
        users_cluster = private_users_index[clusters_index[index]]
        for user, i in enumerate(users_cluster):
            profile_u = np.nonzero(batch_matrix[i, :] * train_mask[user, :])[0]
            u_batch[i, :] = user_update(u_batch[i, :], v_batch[profile_u, :], bias, batch_matrix[i, profile_u])
        f = np.dot(u_batch[users_cluster, :], v_batch.T) + bias
        f = np.maximum(np.minimum(f, 5), 1)
        rmse_imf[index] = calc_rmse(cluster*test_mask, f)
        len_clusters[index] = len(cluster)
        cluster_dens[index] = np.sum(non_zero_matrix(cluster))/ len_clusters[index]


    train_mask, test_mask = build_test_train_masks(batch_matrix)
    pred = user_based_pred(batch_matrix * train_mask)
    rmse_usbas_tot = calc_rmse(batch_matrix * test_mask, pred)
    pass


