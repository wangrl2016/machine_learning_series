from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-2, 2, 400)
    y = x**2
    pyplot.plot(x, y, label='f(x) = x^2')
    pyplot.plot(x, 2 * y + 3, label='m(x) = 2f(x) + 3')
    pyplot.plot(x, 2 * (y + 3), label = 'a(x) = 2(f(x) + 3)')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
