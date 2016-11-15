from util import *
from cluster_build import build_clusters
from hybridRec import hyb_rec_valuation_plot
from expert_set_test import expert_random_subset_plot


if __name__ == "__main__":

    batch_matrix = load_data()
    experts_index, pvt_users_index = expert_base(batch_matrix)
    private_users_matrix = batch_matrix[pvt_users_index, :]
    clusters, clusters_index = build_clusters(private_users_matrix)

    hyb_rec_valuation_plot(batch_matrix, experts_index, clusters)
    expert_random_subset_plot(batch_matrix, experts_index, clusters)

    pass











