import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(-6, 2)
    y = x * x + 4 * x + 3
    pyplot.plot(x, y)
    pyplot.text(0, 3, 'y = x^2 + 4 * x + 3')
    pyplot.axvline(x=-2)
    roots = numpy.roots([1, 4, 3])
    pyplot.scatter(roots, [0, 0])
    for root in roots:
        pyplot.annotate(f'{root}', (root, 0))
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    pyplot.show()

    x = numpy.linspace(-1.2, 1.2)
    for param in [2, 3, 4, 6]:
        y = numpy.power(x, param)
        pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.axis('equal')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    pyplot.show()

    x = numpy.linspace(-4, 4)
    y = -x * x *x + 7 * x * x + 3 * x + 1
    pyplot.plot(x, y)
    pyplot.text(0, 100, 'y = -x^3 + 7 * x^2 + 3 * x + 1')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    pyplot.show()