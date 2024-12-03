import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(-4, 0, 400)
    pyplot.plot(x, x * x + 4 * x + 4, label='f(g(x)) = x^2 + 4x + 4')
    x = numpy.linspace(-2, 2, 400)
    pyplot.plot(x, x * x + 2, label='g(f(x)) = x^2 + 2')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
