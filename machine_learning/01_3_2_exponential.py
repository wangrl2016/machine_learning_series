import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(-4, 4, 800)
    y = numpy.pow(3, x)
    pyplot.plot(x, y, label='y = 3^x')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
