from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-1, 3, 400)
    pyplot.plot(x, (x - 2)**2 + 2, label='f(x) = (x - 2)^2 + 2')
    pyplot.plot(x, -((x - 2)**2 + 2), linestyle='--', label='v(x) = -f(x)')
    pyplot.plot(-x, (-(x - 2))**2 + 2, linestyle='--', label='h(x) = f(-x)')
    pyplot.axvline(x=0)
    pyplot.axhline(y=0)
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
