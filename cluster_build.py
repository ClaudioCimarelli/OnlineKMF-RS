import numpy as np
from sklearn.cluster import KMeans
from util import non_zero_matrix


def build_clusters(users_matrix, n_cluster=10):
    try:
        clusters = np.load('data/clusters.npy')
        clusters_index = np.load('data/clusters_index.npy')
    except:
        clusters, clusters_index = cluster_from(users_matrix, n_cluster)
        np.save('data/clusters', clusters)
        np.save('data/clusters_index', clusters_index)

    return clusters, clusters_index


def cluster_from(users_matrix, n_cluster=10):

    k_means = KMeans(n_clusters=n_cluster, n_init=20, max_iter=350).fit(non_zero_matrix(users_matrix))
    clusters = []
    clusters_index = []

    for i in range(n_cluster):
        ci_index = (np.where(k_means.labels_ == i)[0]).astype('int32')
        clusters_index.append(ci_index)

    random_pick = np.empty([1, 0], dtype='int32')
    for index, cluster_index in enumerate(clusters_index):
        random_rows = np.sort(np.random.choice(len(cluster_index), int(len(cluster_index) * 15 / 100), replace=False))
        random_pick = np.append(random_pick, cluster_index[random_rows])
        clusters_index[index] = np.delete(cluster_index, random_rows)

    split = np.array_split(random_pick, n_cluster)
    for index in range(n_cluster):
        clusters_index[index] = np.sort(np.append(clusters_index[index], split[index]))
        cluster_i = users_matrix[clusters_index[index], :]
        clusters.append(cluster_i)
    return clusters, clusters_index
