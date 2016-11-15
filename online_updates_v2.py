from util import *


def user_update(u_i, v, bias, profile, epochs=30, learn_rate=0.0015, reg_fact=0.06):
    profile = np.reshape(profile, (1, -1)) - bias
    u_i = np.reshape(u_i, (1, -1))
    delta_matrix = np.dot(- 2 * learn_rate * np.eye(v.shape[1]), np.dot(v.T, v)) + (1 - (2 * learn_rate * reg_fact))*np.eye(v.shape[1])
    d = 2 * learn_rate * np.dot(profile, v)
    delta_i = np.ones_like(delta_matrix) + delta_matrix
    delta_matrix_i = np.zeros_like(delta_matrix) + delta_matrix

    for epoch in range(epochs -2):

        delta_matrix_i[...] = np.dot(delta_matrix_i, delta_matrix)

        delta_i[...] += delta_matrix_i

    delta_e = np.dot(delta_matrix_i, delta_matrix)

    u_i[...] = np.dot(u_i, delta_e) + np.dot(d, delta_i)

    np.savetxt('data/delta_e', delta_e, '%.4f')
    np.savetxt('data/delta_sum', delta_i, '%.4f')

    return u_i


def new_user_update(v, bias, profile):
    u_b = np.random.uniform(-0.05, 0.05, len(v[0]))
    return user_update(u_b, v, bias, profile)


