self.velocity = np.zeros_like(self.weights)
self.momentum = np.zeros_like(self.weights)

def adambackprop(self, error, learningRate):
    assert error.shape[1] == self.num_nodes, "shape mismatch"
    grad = error * self.dfunc(self.z)
    delt = self.input.T @ grad
    decay1, decay2, epsilon = 0.9, 0.99, 10**-8
    self.momentum = decay1 * self.momentum + (1-decay1) * delt
    self.velocity = decay2 * self.velocity + (1-decay2) * delt**2
    self.weights -= learningRate * self.momentum / (np.sqrt(self.velocity) + epsilon)
    return grad @ self.weights[:-1].T

def rmsbackprop(self, error, learningRate):
    assert error.shape[1] == self.num_nodes, "shape mismatch"
    grad = error * self.dfunc(self.z)
    delt = self.input.T @ grad
    decay, epsilon = 0.95, 10**-8
    self.velocity = decay * self.velocity + (1-decay) * delt**2
    self.weights -= learningRate * delt / (np.sqrt(self.velocity) + epsilon)
    return grad @ self.weights[:-1].T
