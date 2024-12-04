from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-1, 1, 200)
    pyplot.plot(x, x**2, label = 'f(x) = x^2')
    pyplot.plot(x, (2 * x)**2, label = 'c(x) = f(2x)')
    pyplot.plot(x, (x / 2)**2, label = 's(x) = f(x/2)')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
