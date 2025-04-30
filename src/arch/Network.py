from .DenseLayer import DenseLayer
import numpy as np

class Network:
    """
        Whole Neural Network with Parameters and Layers
    """

    def __init__(self, num_input):
        self.num_input = num_input
        self.layers = []

    def addLayer(self, num_nodes, activation='sigmoid'):
        inputs = self.layers[-1].num_nodes if len(self.layers) else self.num_input
        self.layers.append(DenseLayer(inputs, num_nodes, act=activation))

    def fit(self, train, val, loss='categoricalCrossentropy',
                learningRate=0.01, batch_size=8, epochs=84):

        features = train.columns[2:]
        train[features] = (train[features] - train[features].mean()) / train[features].std()
        train[1] = train[1].map({'M': 1, 'B': 0})
        for _ in range(100):
            for i in range(0, len(train), batch_size):
        
                output = train.iloc[i:i+batch_size]
                for layer in self.layers:
                    output = layer.calculate(output)
                
                pred = np.array(train.iloc[i:i+batch_size, 1])
                for i in reversed(self.layers):
                    pred = layer.backprop(pred)
        
        
        for i in range(0, len(train), batch_size):
            output = train.iloc[i:i+batch_size]
            for layer in self.layers:
                output = layer.calculate(output)
            np.set_printoptions(suppress=True, linewidth=200)
            print(np.transpose(output))
