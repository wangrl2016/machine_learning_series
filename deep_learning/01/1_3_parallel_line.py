from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-1, 3)
    pyplot.plot(x, -2 * x + 6, label='f(x) = -2x + 6')
    pyplot.plot(x, -2 * x - 4, label='f(x) = -2x - 4')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
