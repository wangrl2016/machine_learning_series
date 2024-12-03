from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-2, 2, 400)
    pyplot.plot(x, numpy.abs(x), label = 'f(x) = |x|')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.axis('equal')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
