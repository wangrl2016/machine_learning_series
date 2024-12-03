import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(-2, 2, 400)
    pyplot.plot(x, x**2, label='y = x^2')
    x = numpy.linspace(0, 4, 400)
    pyplot.plot(x, (x - 2)**2, label='y = (x - 2)^2')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.axis('equal')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
