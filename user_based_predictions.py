import numpy as np
from cosinesim import cos_sim
from util import non_zero_matrix, calc_rmse


def user_based_pred(users_matrix):
    sim = cos_sim(users_matrix)
    nz_us = non_zero_matrix(users_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_ratings = np.nan_to_num(np.sum(users_matrix, axis=1)/ np.sum(nz_us, axis=1)).reshape((-1, 1))
    deviations = (users_matrix - avg_ratings)*nz_us
    num = np.dot(sim, deviations)
    den = np.dot(sim, nz_us)
    with np.errstate(divide='ignore', invalid='ignore'):
        base_pred = np.nan_to_num(np.divide(num, den))
    pred = avg_ratings + base_pred
    rmse = calc_rmse(users_matrix, pred)
    return np.maximum(np.minimum(pred, 5), 1)



