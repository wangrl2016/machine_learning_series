import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(-3, 3, 600)
    y = numpy.sign(x) * numpy.power(numpy.abs(x), 1/3)
    pyplot.plot(x, y, label='y=x^1/3')
    y = y + 1
    pyplot.plot(x, y, label='y=x^1/3 + 1')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

