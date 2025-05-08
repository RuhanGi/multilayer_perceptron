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



def main():
    accs = []
    k = 400
    bs, ba, bn = 0,0, None
    offset = np.random.randint(10**5)
    for i in range(offset, offset+k):
        acc, n, met = run(train, train_out, val, val_out, seed=i, opt='adam')
        print(GRAY + f"\rModel: {i-offset}/{k}", end="")
        accs.append(acc)
        if acc > ba:
            bs, ba, bn = i, acc, n

    print(BLUE + f"\rBest Accuracy: {PURPLE}{ba*100:.4f}%{BLUE} by Seed {i}" + RESET)
    print(f"{GREEN}{k}-Fold Acc: {PURPLE}{np.average(accs)*100:.4f}%{RESET}")

if __name__ == "__main__":
    main()
