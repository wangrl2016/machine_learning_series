from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-3, 3, 600)
    pyplot.plot(x, x**4 - x**3 - 4 * x**2 + 4 * x, label='f(x) = x^4 - x^3 - 4x^2 + 4x')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
