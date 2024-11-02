import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(0, 4 * numpy.pi, 500)
    y = numpy.sin(x)
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.06)
    pyplot.show()

    y = numpy.cos(x)
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.06)
    pyplot.show()
