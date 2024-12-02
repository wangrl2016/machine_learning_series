from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(0, 1)
    pyplot.plot(x, x / 2 + 2, label='f(x) = x/2 + 2')
    pyplot.plot(x, -2 * x + 3, label='f(x) = -2x + 3')
    pyplot.grid(True)
    pyplot.axis('equal')
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
