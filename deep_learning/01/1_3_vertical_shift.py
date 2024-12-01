from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-4, 4)
    pyplot.plot(x, x + 4, label='f(x) = x + 4')
    pyplot.plot(x, x + 2, label='f(x) = x + 2')
    pyplot.plot(x, x, label='f(x) = x')
    pyplot.plot(x, x - 2, label='f(x) = x - 2')
    pyplot.plot(x, x - 4, label='f(x) = x - 4')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
