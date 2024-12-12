from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    x = numpy.linspace(-1, 1, 200)
    pyplot.plot(x, x**2, label='f(x) = x^2')
    pyplot.plot(x, x**3, label='f(x) = x^3')
    pyplot.plot(x, x**4, label='f(x) = x^4')
    pyplot.plot(x, x**5, label='f(x) = x^5')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.axis('equal')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
