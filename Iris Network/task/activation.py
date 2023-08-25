import numpy as np


def sigmoid(x):
    # implement the logistic sigmoid function based on the formula in the task description
    # you may need the numpy.exp function
    return 1 / (1 + np.exp(-x))
