import numpy as np


def cos_sim(users_matrix):
    norm = np.linalg.norm(users_matrix, axis=1).reshape((-1,1))
    with np.errstate(divide='ignore', invalid='ignore'):
        div = np.nan_to_num(np.divide(users_matrix, norm))
    sim = np.dot(div, div.T)
    return sim
