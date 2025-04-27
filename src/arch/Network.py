from .DenseLayer import DenseLayer


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

    
        output = train.iloc[0][2:]
        for i in self.layers:
            output = i.calculate(output)
        print(output)


        while True:
            self.layers[-1].backprop(output)
            output = train.iloc[0][2:]
            for i in self.layers:
                output = i.calculate(output)
            print(output)
        return output
        