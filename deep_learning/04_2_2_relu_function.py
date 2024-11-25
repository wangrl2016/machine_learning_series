import numpy
from matplotlib import pyplot

def relu(x):
    return numpy.maximum(0, x)

def deriv_relu(x):
    return numpy.where(x > 0, 1, 0)

if __name__ == '__main__':
    x = numpy.linspace(-5, 5, 1000)
    pyplot.plot(x, relu(x), label='f(x) = x > 0 ? x : 0')
    pyplot.plot(x, deriv_relu(x), label='f\'(x) = x > 0 ? 1 : 0')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
