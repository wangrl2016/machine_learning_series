from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-5, 10, 1500)
    pyplot.plot(x, x**3 - 6 * x**2 - 15 * x + 20, label='y = x^3 - 6x^2 - 15x + 20')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
