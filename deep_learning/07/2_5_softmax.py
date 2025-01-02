import numpy

class Softmax:
    # A Standard fully-connected layer with softmax activation.
    def __init__(self, input_len, nodes):
        rng = numpy.random.default_rng(0)
        self.weights = rng.random((input_len, nodes)) / input_len
        self.biases = numpy.zeros(nodes)
    
    def forward(self, input):
        # Performs a forward pass of the softmax layer using the given input.
        # Returns a 1d numpy array containing the respective probability values.
        input = input.flatten()
        totals = numpy.dot(input, self.weights) + self.biases
        exp = numpy.exp(totals)
        return exp / numpy.sum(exp, axis=0)
