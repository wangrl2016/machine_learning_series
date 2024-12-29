import numpy
from matplotlib import pyplot

def tanh(x):
    return (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))

def deriv_tanh(x):
    return 1 - tanh(x)**2

if __name__ == '__main__':
    x = numpy.linspace(-5, 5, 1000)
    pyplot.plot(x, numpy.tanh(x) + 1, label='f(x) = numpy.tanh(x) + 1')
    pyplot.plot(x, tanh(x), label='g(x) = (e^x - e^(-x)) / (e^x + e^(-x))')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
    pyplot.plot(x, tanh(x), label='f(x) = (e^x - e^(-x)) / (e^x + e^(-x))')
    pyplot.plot(x, deriv_tanh(x), label='f\'(x) = 1 - tanh(x)^2')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

    print('The derivative at x = 2 is', numpy.round(deriv_tanh(2), 4))
