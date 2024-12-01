from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-1, 4, 500)
    (x1, y1) = (0, 1)
    (x2, y2) = (3, 2)
    m = (y2 - y1) / (x2 - x1)
    y = m * (x - x1) + y1
    pyplot.plot(x, y)
    pyplot.scatter([x1, x2], [y1, y2], color='red')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
