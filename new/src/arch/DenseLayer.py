from dataclasses import dataclass, 
import numpy as np
import actfunc

# ! Might Delete if passing in learningRate backprop
# self.velocity = np.zeros_like(self.weights)
# self.momentum = np.zeros_like(self.weights)

class DenseLayer:
    """
        Stores a single Layer of nodes
        
        Self Attributes:
        func: activation function
        dfunc: derivative of activation function
    
        weights: input+bias for each node     (num_input+1, num_nodes)
          input: input+bias for each sample   (samples, num_input+1)
              z: w•i for each node and sample (samples, num_nodes)
         output: applying activation function (samples, num_nodes)
    """

    def __init__(self, num_input, num_nodes, act):
        limit = np.sqrt(6 / num_input)
        self.weights = np.random.uniform(-limit, limit, (num_input+1, num_nodes))
        self.func, self.dfunc = actfunc.getFunc(act)

    def calculate(self, inputs):
        assert inputs.shape[1]+1 == weights.shape[0], "shape mismatch"
        self.input = np.hstack([inputs, np.ones((inputs.shape[0], 1))])
        self.z = self.input @ self.weights
        self.output = self.func(self.z)
        return self.output

    def backprop(self, error, learningRate):
        """
          error: dL/df() (samples, num_nodes)
          dfunc: df()/dz (samples, num_nodes)
          input: dz/dw   (samples, num_input+1)

          delta: dL/dz   (samples, num_nodes)
           grad: dL/dw   (num_input+1, num_nodes)
         
        weights: dz/dy1  (num_input+1, num_nodes)
          input: input+bias for each sample   (samples, num_input+1)
        """
        assert error.shape == self.z.shape, "shape mismatch"
        delta = error * self.dfunc(self.z)
        grad = self.input.T @ delta
        self.weights -= learningRate * grad
        return delta @ self.weights[:-1].T
