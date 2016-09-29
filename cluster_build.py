import numpy as np
from sklearn.cluster import KMeans
from util import non_zero_matrix


def build_clusters(users_matrix):
    try:
        clusters = np.load('data/clusters.npy')
        clusters_index = np.load('data/clusters_index.npy')
    except:
        clusters, clusters_index = cluster_from(users_matrix)
        np.save('data/clusters', clusters)
        np.save('data/clusters_index', clusters_index)

    return clusters, clusters_index


def cluster_from(users_matrix):
    k_means = KMeans(n_clusters=10, n_init= 20).fit(non_zero_matrix(users_matrix))
    clusters = []
    clusters_index = []
    for i in range(10):
        ci_index = np.where(k_means.labels_ == i)[0]
        clusters_index.append(ci_index)
        clusters.append(users_matrix[ci_index, :])

    random_pick = np.empty([1, 0], dtype='int64')
    for index, cluster_index in enumerate(clusters_index):
        random_rows = np.sort(np.random.choice(len(cluster_index), int(len(cluster_index) * 4 / 10), replace=False))
        clusters[index] = np.delete(clusters[index], random_rows, 0)
        random_pick = np.append(random_pick, cluster_index[random_rows])
        clusters_index[index] = np.delete(cluster_index, random_rows, 0)

    split = np.array_split(random_pick, 10)
    for index, cluster in enumerate(clusters):
        clusters[index] = np.append(cluster, users_matrix[split[index], :], 0)
        clusters_index[index] = np.append(clusters_index[index], split[index])

    return clusters, clusters_index
