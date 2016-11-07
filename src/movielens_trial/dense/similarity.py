import numpy as np


def cosine_sim(ui_mat, epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    sim = ui_mat.dot(ui_mat.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
