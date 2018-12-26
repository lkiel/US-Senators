import numpy as np


def error(truth, estimation):
    return np.linalg.norm(truth-estimation) / np.linalg.norm(truth)


def accuracy(truth, estimation):
    return (truth == estimation).mean()