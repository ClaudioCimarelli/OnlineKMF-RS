from util import load_data, non_zero_matrix, np
from batch_MF import train
from cluster_build import build_clusters
from user_based_predictions import user_based_pred



if __name__ == "__main__":

    batch_matrix = load_data()
    nz_batch = non_zero_matrix(batch_matrix)
    ratings_count = np.sum(nz_batch, axis=1)
    sort_by_ratings_index = np.argsort(ratings_count, kind='heapsort')
    experts_index = sort_by_ratings_index[-4000:]
    private_users_index = sort_by_ratings_index[:-4000]
    experts_matrix = batch_matrix[experts_index, :]
    private_users_matrix = batch_matrix[private_users_index, :]

    mask = np.ones_like(experts_matrix)
    N = len(batch_matrix)
    M = len(batch_matrix[0])
    K = 40
    u_batch, v_batch = train(experts_matrix, mask, N, M, K, suffix_name='experts')

    clusters, clusters_index = build_clusters(private_users_matrix)

    for cluster in clusters:
        pred = user_based_pred(cluster)