import numpy as np

def sigmoid(x):
    out = np.empty_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)
    return out

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return 2 * sigmoid(2 * x) - 1

def dtanh(x):
    return 4 * dsigmoid(2 * x)

def ReLU(x, r=0):
    return np.maximum(r * x, x)

def dReLU(x, r=0):
    return np.where(x > 0, 1, r)

def ELU(x, alpha=1.67326, lam=1.0507):
    return lam * np.maximum(alpha * (np.exp(x) - 1), x)

def dELU(x, alpha=1.67326, lam=1.0507):
    return lam * np.where(x > 0, 1, alpha * np.exp(x) * (np.exp(x) - 1))

def linear(z):
    return z

def dlinear(z):
    return np.ones_like(z)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)

def softplus(x):
    return x if x > 20 else np.log(1 + np.exp(x))

def dsoftplus(x):
    return 1 if x > 20 else 1 - sigmoid(-x)

def getFunc(act):
    try:
        funcy = {
            'sigmoid' : (sigmoid, dsigmoid),
            'tanh' : (tanh, dtanh),
            'ReLU' : (ReLU, dReLU),
            'ELU' : (ELU, dELU),
            'softmax' : (softmax, dlinear),
            'softplus' : (softplus, dsoftplus)
        }
        return funcy[act]
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)