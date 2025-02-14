from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-4, 4, 800)
    y = numpy.exp(-(x**2) / 2)
    pyplot.plot(x, y, label='f(x) = e^(-x*x/2)')
    pyplot.plot(x, 2 * y, label='s(x) = 2f(x)')
    pyplot.plot(x, y / 2, label='c(x) = f(x) / 2')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
