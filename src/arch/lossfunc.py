import numpy as np

def binaryCrossEntropy(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    return -np.mean()

def crossEntropy(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    per_sample = -np.sum(one_hot * np.log(probs), axis=1)
    return np.mean(per_sample)

def meanSquare(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    return np.mean(np.sum((probs - one_hot)**2, axis=1))

def rootMeanSquare(probs, one_hot):
    assert probs.shape == one_hot.shape, "shape mismatch"
    return np.sqrt(np.mean(np.sum((probs - one_hot)**2, axis=1)))

def getLoss(loss):
    funcy = {
        'binaryCrossEntropy' : binaryCrossEntropy,
        'crossEntropy' : crossEntropy,
        'meanSquare' : meanSquare,
        'rootMeanSquare' : rootMeanSquare
    }
    return funcy[loss]
