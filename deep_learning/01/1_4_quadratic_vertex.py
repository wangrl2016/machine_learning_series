from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-5, 1)
    pyplot.plot(x, -3 * (x + 2)**2 + 4, label='y = -3(x+2)^2 + 4')
    pyplot.axvline(x=-2, linestyle='--', label='x = -2')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
