from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-5, 1)
    y = x * x + 4 * x + 3
    pyplot.plot(x, y, label='y = x^2 + 4x + 3')
    roots = numpy.roots([1, 4, 3])
    pyplot.scatter(roots, [0, 0])
    pyplot.axvline(x=-2, linestyle='--', label='x = -2')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
