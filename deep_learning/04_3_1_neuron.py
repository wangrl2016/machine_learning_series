import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = numpy.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

if __name__ == '__main__':
    weights = numpy.array([0, 1])
    bias = 4
    neuron = Neuron(weights, bias)
    x = numpy.array([2, 3])
    # 0.9990889488055994
    print(neuron.feedforward(x))
