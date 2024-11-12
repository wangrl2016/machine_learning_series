import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(-4, 6, 1000)
    pyplot.plot(x, (2*x-2)**2, label='f(x)=(2x-2)^2')
    pyplot.scatter([4], [36], color='red', label='(4, 36)')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
