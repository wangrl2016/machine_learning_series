from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    (x1, y1) = (3, 0)
    pyplot.scatter(x1, y1)
    x = numpy.linspace(-2, 0)
    pyplot.plot(x, 3 * x + 4, label='f(x) = 3x + 4')
    x = numpy.linspace(-3, 4)
    pyplot.plot(x, -x / 3 + 1, label='f(x) = -x/3 + 1')
    pyplot.legend()
    pyplot.axis('equal')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
