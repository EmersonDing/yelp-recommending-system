import numpy as np


def cosine_sim(ui_mat, epsilon=1e-9):
    sim = ui_mat.dot(ui_mat.T)
    sim = sim.toarray() + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
