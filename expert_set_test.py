from util import *
import matplotlib.pyplot as plt
from batch_MF import train
from user_based_predictions import user_based_pred
from hybridRec import hybrid_rec, imf_pred_cluster


def expert_random_subset_plot(batch_matrix,experts_index, clusters):

    train_mask, test_mask = build_test_train_masks(clusters[0])

    random_experts = np.random.permutation(experts_index)

    rmse_comb = np.zeros((3, 8))

    for i, usr_num in enumerate(np.linspace(500, 4000, 8)):
        experts_subset_index = np.sort(random_experts[-int(usr_num):])

        ###User based predictions
        # user_based_pool = np.append(batch_matrix[experts_subset_index, :], clusters[0] * train_mask, axis=0)
        ub_pred = user_based_pred(clusters[0] * train_mask)

        mask = np.zeros_like(batch_matrix)
        mask[experts_subset_index, :] = 1

        N = len(batch_matrix)
        M = len(batch_matrix[0])
        K = 10
        u_batch, v_batch = train(batch_matrix * mask, N, M, K, suffix_name='experts_random' + str(int(usr_num)))
        nz_ratings = non_zero_matrix(batch_matrix * mask)
        bias = np.sum(batch_matrix * mask) / np.sum(nz_ratings)
        imf_pred = imf_pred_cluster(clusters[0] * train_mask, v_batch, bias)

        result = hybrid_rec(imf_pred, ub_pred, alpha=0.8)

        rmse_comb[0, i] = calc_rmse(clusters[0] * test_mask, result)
        rmse_comb[1, i] = calc_rmse(clusters[0] * test_mask, imf_pred)
        rmse_comb[2, i] = calc_rmse(clusters[0] * test_mask, ub_pred)

    plt.plot(np.linspace(500, 4000, 8), rmse_comb[0], label='hybrid')
    plt.plot(np.linspace(500, 4000, 8), rmse_comb[1], label='imf')
    plt.plot(np.linspace(500, 4000, 8), rmse_comb[2], label='user based')
    plt.xlabel('random experts length')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(ncol=3)
    plt.show()