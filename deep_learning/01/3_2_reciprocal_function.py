from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(0.1, 3, 300)
    pyplot.plot(x,  1 / x, color='blue', label='f(x) = 1/x')
    x = numpy.linspace(-3, -0.1, 300)
    pyplot.plot(x, 1 /x, color='blue')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.06)
    pyplot.show()
