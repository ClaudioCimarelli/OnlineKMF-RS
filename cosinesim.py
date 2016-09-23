import numpy as np


def cos_sim(users_matrix):
    norm = np.linalg.norm(users_matrix, axis=1)
    norm = norm.reshape((len(norm),1))
    div = np.divide(users_matrix, norm)
    sim = np.dot(div, div.T)
    return sim