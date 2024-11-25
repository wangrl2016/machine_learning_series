import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(0, 2, 200)
    pyplot.plot(x, x * x, label='y = x^2')
    x = numpy.linspace(0, 4, 400)
    pyplot.plot(x, numpy.power(x, 1/2), label='y = x^1/2')
    x = numpy.linspace(0, 3, 300)
    pyplot.plot(x, x, label='y = x', linestyle='--')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.axis('equal')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
