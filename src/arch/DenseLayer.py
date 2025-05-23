from .actfunc import getFunc
import numpy as np

class DenseLayer:
    """
    Stores a single Layer of nodes
  
      Self Attributes:
      func: activation function
      dfunc: derivative of activation function
      weights: random initial weights (num_input+1, num_nodes)
    """

    def __init__(self, num_input, num_nodes, act):
        assert isinstance(num_input, int), "integers only"
        assert isinstance(num_nodes, int), "integers only"

        limit = np.sqrt(6 / num_input)
        self.weights = np.random.uniform(-limit, limit, (num_input+1, num_nodes))
        self.func, self.dfunc = getFunc(act)

        self.velocity = np.zeros_like(self.weights)
        self.momentum = np.zeros_like(self.weights)

    def forward(self, inputs):
      """
      Feedforward

        weights: input+bias for each node     (num_input+1, num_nodes)
          input: input+bias for each sample   (samples, num_input+1)
              z: w•i for each node and sample (samples, num_nodes)
         output: applying activation function (samples, num_nodes)
      """
      assert inputs.shape[1]+1 == self.weights.shape[0], "shape mismatch"

      self.input = np.hstack([inputs, np.ones((inputs.shape[0], 1))])
      self.z = self.input @ self.weights
      self.output = self.func(self.z)
      return self.output

    def backprop(self, error, learningRate, opt, t):
      """
        Backpropate using the loss and update weights

          error: dL/dy  (samples, num_nodes)
          dfunc: dy/dz  (samples, num_nodes)
          input: dz/dw  (samples, num_input+1)

          delta: dL/dz  (samples, num_nodes)
           grad: dL/dw  (num_input+1, num_nodes)
          wghts: dz/dy1 (num_input+1, num_nodes)
          retrn: dL/dy1 (samples, num_input)
      """
      assert error.shape == self.z.shape, "shape mismatch"
      delta = error * self.dfunc(self.z)
      grad = self.input.T @ delta

      if opt == 'rmsprop':
          decay2, epsilon = 0.99, 1e-8
          self.velocity = decay2 * self.velocity + (1 - decay2) * grad**2
          update = grad / (np.sqrt(self.velocity) + epsilon)
          self.weights -= learningRate * update
      elif opt == 'adam':
          decay1, decay2, epsilon = 0.9, 0.999, 1e-8
          self.momentum = decay1 * self.momentum + (1 - decay1) * grad
          self.velocity = decay2 * self.velocity + (1 - decay2) * grad**2
          m_hat = self.momentum / (1 - decay1 ** t)
          v_hat = self.velocity / (1 - decay2 ** t)
          update = m_hat / (np.sqrt(v_hat) + epsilon)
          self.weights -= learningRate * update
      else:
          self.weights -= learningRate * grad

      return delta @ self.weights[:-1].T
