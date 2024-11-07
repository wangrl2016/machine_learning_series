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

class OurNeuralNetwork:
    def __init__(self):
        weights = numpy.array([0, 1])
        bias = 0
        
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
    
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))
        return out_o1

if __name__ == '__main__':
    network = OurNeuralNetwork()
    x = numpy.array([2, 3])
    # 0.7216325609518421
    print(network.feedforward(x))
