import numpy

def softmax(z):
    return numpy.exp(z) / numpy.sum(numpy.exp(z))

def deriv_softmax(z):
    s = softmax(z)
    jacobian_matrix = numpy.diag(s) - numpy.outer(s, s)
    return jacobian_matrix

if __name__ == '__main__':
    z = numpy.array([2.0, 1.0, 0.1])
    print(numpy.round(deriv_softmax(z), 4))
