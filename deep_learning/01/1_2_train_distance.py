from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(0, 5)
    y = 83 * x + 250
    pyplot.plot(x, y, label='D(t) = 83t + 250')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()