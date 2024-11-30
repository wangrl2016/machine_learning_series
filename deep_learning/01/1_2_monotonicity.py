from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(0, 2, 200)
    pyplot.plot(x, 2 * x + 3, label='Increasing Function')
    pyplot.plot(x, -2 * x + 3, label='Decreasing Function')
    pyplot.plot(x, 0 * x + 3, label='Constant Function')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
