import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    fig, ax = pyplot.subplots()
    ax.axis('equal')
    x = numpy.linspace(-4, 4)
    y = 3 * x + 4
    pyplot.plot(x, y)
    y = -1/3 * x + 1
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.show()
