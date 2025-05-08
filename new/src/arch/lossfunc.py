import numpy as np


def crossEntropy(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    return np.mean(-np.sum(one_hot * np.log(probs), axis=1))

def dcrossEntropy(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    return -one_hot / probs

def meanSquare(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    return np.mean(np.sum((probs - one_hot)**2, axis=1))

def dmeanSquare(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    return probs - one_hot

def rootMeanSquare(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    return np.sqrt(np.mean(np.sum((probs - one_hot)**2, axis=1)))

def drootMeanSquare(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    return probs - one_hot

def getLoss(loss):
    funcy = {
        'crossEntropy' : (crossEntropy, dcrossEntropy),
        'meanSquare' : (meanSquare, dmeanSquare),
        'rootMeanSquare' : (rootMeanSquare, drootMeanSquare)
    }
    return funcy[loss]