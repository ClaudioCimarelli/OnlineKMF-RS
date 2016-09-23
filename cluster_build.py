import numpy as np
from sklearn.cluster import KMeans
from util import non_zero_matrix


def build_clusters(users_matrix):
    try:
        clusters = np.load('data/clusters.npy')
        clusters_index = np.load('data/clusters_index.npy')
    except:
        k_means = KMeans(n_clusters=10).fit(non_zero_matrix(users_matrix))
        clusters = []
        clusters_index = []
        for i in range(10):
            ci_index = np.where(k_means.labels_ == i)[0]
            clusters_index.append(ci_index)
            clusters.append(users_matrix[ci_index, :])
        np.save('data/clusters', clusters)
        np.save('data/clusters_index', clusters_index)

    return clusters, clusters_index
