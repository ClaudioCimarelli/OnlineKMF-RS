from util import *
import matplotlib.pyplot as plt
from user_based_predictions import user_based_pred
from online_updates_v2 import new_user_update
from batch_MF import train


def hyb_rec_valuation_plot(batch_matrix, experts_index, clusters):


    mask = np.zeros_like(batch_matrix)
    mask[experts_index, :] = 1

    N = len(batch_matrix)
    M = len(batch_matrix[0])
    K = 10
    u_batch, v_batch = train(batch_matrix * mask, N, M, K, suffix_name='experts')
    nz_ratings = non_zero_matrix(batch_matrix * mask)
    bias = np.sum(batch_matrix * mask) / np.sum(nz_ratings)

    #  Plotting data arrays
    rmse_user_based = np.zeros((10, 1))
    rmse_imf = np.zeros((10, 1))
    clust_len = np.zeros((10, 1))
    clust_dens = np.zeros((10, 1))
    rmse_hyb = np.zeros((10, 11))  # array with all values of rmse with alpha from 0 to 1(step length 0.1)
    rmse_hyb_best = np.zeros((10, 2))  # array with best rmse and value of alpha to obtain it

    for index, cluster in enumerate(clusters):
        ###train and test masks per each cluster
        train_mask, test_mask = build_test_train_masks(cluster)

        # ###User based predictions
        # user_based_pool = np.append(batch_matrix[experts_index, :], cluster * train_mask, axis=0)
        # ub_pred = user_based_pred(user_based_pool)[len(experts_index):]
        ub_pred = user_based_pred(cluster * train_mask)
        rmse_user_based[index] = calc_rmse(cluster * test_mask, ub_pred)

        ###IMF predictions
        imf_pred = imf_pred_cluster(cluster*train_mask, v_batch, bias)

        rmse_imf[index] = calc_rmse(cluster * test_mask, imf_pred)

        ### Combined predictions for alpha form 0 to 1 with steps of 0.1
        hyb_rmse_alpha = np.zeros(11)
        for i, alpha in enumerate(np.linspace(0, 1, 11)):
            hyb_pred = alpha * imf_pred + (1 - alpha) * ub_pred
            hyb_rmse_alpha[i] = calc_rmse(cluster * test_mask, hyb_pred)
        rmse_hyb[index] = hyb_rmse_alpha

        ### find best alpha and rmse
        rmse_hyb_best[index, 0] = np.linspace(0, 1, 11)[np.argmin(hyb_rmse_alpha)]
        rmse_hyb_best[index, 1] = np.min(hyb_rmse_alpha)

        ### save cluster density and length
        clust_dens[index] = 100 * (np.sum(non_zero_matrix(cluster)) / cluster.size)
        clust_len[index] = len(cluster)

        ###prepare the plot for the hybrid rec varying alpha
        if index > 4:
            linestyle = "--"
        else:
            linestyle = "-"
        plt.plot(np.linspace(0, 1, 11), hyb_rmse_alpha, label='Group ' + str(index + 1), ls=linestyle)

    plt.plot(rmse_hyb_best[:, 0], rmse_hyb_best[:, 1], 'kx', label="min RMSE")

    plt.xlabel('alpha')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(ncol=5)
    plt.savefig('plots/rmse_combined_cluster.png', bbox_inches='tight')
    plt.show()

    ###save data for latex tables
    np.savetxt('data/rmse_hyb', rmse_hyb, '%.4f', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/rmse_ub', rmse_user_based.T, '%.4f', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/rmse_imf', rmse_imf.T, '%.4f', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/clus_len', clust_len.T, '%.d', delimiter=' &', newline='\\\\ \hline \n')
    np.savetxt('data/clus_dens', clust_dens.T, '%.4f', delimiter=' &', newline='\\\\ \hline \n')
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

    return imf_pred
