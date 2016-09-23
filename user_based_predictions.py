import numpy as np
from cosinesim import cos_sim
from util import non_zero_matrix


def user_based_pred(users_matrix):
    sim = cos_sim(users_matrix)
    nz_us = non_zero_matrix(users_matrix)
    avg_ratings = (np.sum(users_matrix, axis=1)/ np.sum(nz_us, axis=1)).reshape((-1, 1))
    deviations = (users_matrix - avg_ratings)*nz_us
    pred = avg_ratings + (np.dot(sim, deviations)/ np.sum(sim, axis=1))


