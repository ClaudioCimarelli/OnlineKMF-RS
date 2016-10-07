from util import *
import matplotlib.pyplot as plt
from user_based_predictions import user_based_pred
from online_updates import user_update, new_user_update
from batch_MF import train
from cluster_build import build_clusters


def hybrid_rec_test(batch_matrix,experts_index, private_users_index):

    private_users_matrix = batch_matrix[private_users_index, :]
    clusters, clusters_index = build_clusters(private_users_matrix)

    mask = np.zeros_like(batch_matrix)
    mask[experts_index, :] = 1

    N = len(batch_matrix)
    M = len(batch_matrix[0])
    K = 40
    u_batch, v_batch = train(batch_matrix * mask, N, M, K, suffix_name='experts4000')
    nz_ratings = non_zero_matrix(batch_matrix * mask)
    bias = np.sum(batch_matrix * mask) / np.sum(nz_ratings)
    rmse_user_based = np.zeros((10, 1))
    rmse_imf = np.zeros((10, 1))
    clust_len = np.zeros((10, 1))
    clust_dens = np.zeros((10, 1))
    rmse_comb = np.zeros((10, 31))
    rmse_comb_best = np.zeros((10, 2))

    for index, cluster in enumerate(clusters):
        ###train and test set per each cluster
        train_mask, test_mask = build_test_train_masks(cluster)

        ###User based predictions
        user_based_pool = np.append(batch_matrix[experts_index, :], cluster * train_mask, axis=0)
        ub_pred = user_based_pred(user_based_pool)[len(experts_index):]
        # ub_pred = user_based_pred(cluster * train_mask)
        # ub_pred = np.maximum(np.minimum(ub_pred, 5), 1)
        # rmse_user_based[index] = calc_rmse(cluster * test_mask, ub_pred)

        ###IMF predictions
        users_cluster = private_users_index[clusters_index[index]]
        for user, i in enumerate(users_cluster):
            profile_u = np.nonzero(batch_matrix[i, :] * train_mask[user, :])[0]
            u_batch[i, :] = user_update(u_batch[i, :], v_batch[profile_u, :], bias, batch_matrix[i, profile_u])
        imf_pred = np.dot(u_batch[users_cluster, :], v_batch.T) + bias
        imf_pred = np.maximum(np.minimum(imf_pred, 5), 1)
        rmse_imf[index] = calc_rmse(cluster * test_mask, imf_pred)

        ### Combined predictions
        rmse_comb_alpha = np.zeros(31)
        for i, alpha in enumerate(np.linspace(0, 1, 31)):
            comb_pred = alpha * imf_pred + (1 - alpha) * ub_pred
            rmse_comb_alpha[i] = calc_rmse(cluster * test_mask, comb_pred)
        rmse_comb[index] = rmse_comb_alpha

        rmse_comb_best[index, 0] = np.linspace(0, 1, 31)[np.argmin(rmse_comb_alpha)]
        rmse_comb_best[index, 1] = np.min(rmse_comb_alpha)

        clust_dens[index] = 100 * (np.sum(non_zero_matrix(cluster)) / cluster.size)
        clust_len[index] = len(cluster)

        if index > 4:
            linestyle = "--"
        else:
            linestyle = "-"
        plt.plot(np.linspace(0, 1, 31), rmse_comb_alpha, label='Group ' + str(index + 1), ls=linestyle)
        # plt.plot(rmse_comb_best[index,0], rmse_comb_best[index,1], 'kx')

    plt.plot(rmse_comb_best[:, 0], rmse_comb_best[:, 1], 'kx', label="min RMSE")

    plt.xlabel('alpha')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(ncol=5)
    plt.savefig('plots/rmse_combined_cluster.png', bbox_inches='tight')
    plt.show()

    np.savetxt('data/rmse_comb', rmse_comb, '%.4f', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/rmse_ub', rmse_user_based.T, '%.4f', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/rmse_imf', rmse_imf.T, '%.4f', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/clus_len', clust_len.T, '%.d', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/clus_dens', clust_dens.T, '%.4f', delimiter=' &', newline='\\\\ \hline \n')

    train_mask, test_mask = build_test_train_masks(batch_matrix)
    ub_pred = user_based_pred(batch_matrix * train_mask)
    rmse_usbas_tot = calc_rmse(batch_matrix * test_mask, ub_pred)
    pass


def hybrid_rec(imf_pred, ub_pred, alpha=0.8):
    ### Combined predictions
    comb_pred = alpha * imf_pred + (1 - alpha) * ub_pred

    return comb_pred

def imf_pred_cluster(cluster, v_batch, bias):
    u_cluster = np.zeros((len(cluster), len(v_batch[0])))
    ###IMF predictions
    for user in range(len(cluster)):
        profile_u = np.nonzero(cluster[user, :])[0]
        u_i = new_user_update(v_batch[profile_u, :], bias, cluster[user, profile_u])
        u_cluster[user] = u_i
    imf_pred = np.dot(u_cluster, v_batch.T) + bias
    imf_pred = np.maximum(np.minimum(imf_pred, 5), 1)

    return u_cluster,imf_pred
