import numpy
from matplotlib import pyplot

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

if __name__ == '__main__':
    x = numpy.linspace(-10, 10, 400)
    pyplot.plot(x, sigmoid(x), label='f(x) = 1 / (1 + e^-x)')
    pyplot.plot(x, deriv_sigmoid(x), label = 'f\'(x) = f(x) * (1 - f(x))')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
