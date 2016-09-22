from util import load_data, non_zero_matrix, np
from sklearn.cluster import KMeans
from batch_MF import train

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

    kmeans = KMeans(n_clusters=10).fit(non_zero_matrix(private_users_matrix))
    clusters = []
    clusters_index = []
    for i in range(10):
        ci_index = np.where(kmeans.labels_ == i)[0]
        clusters_index.append(ci_index)
        clusters.append(private_users_matrix[ci_index, :])

    pass