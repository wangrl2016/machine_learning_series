import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    fig, ax = pyplot.subplots()
    ax.axis('equal')
    pyplot.scatter(3, 0)
    pyplot.annotate('(3, 0)', (3, 0))
    x = numpy.linspace(-4, 4)
    y = 3 * x + 4
    pyplot.annotate('y = 3 * x + 4', (3, 13))
    pyplot.plot(x, y)
    y = -1/3 * x + 1
    pyplot.annotate('y = -1/3 * x + 1', (-6, 3))
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
