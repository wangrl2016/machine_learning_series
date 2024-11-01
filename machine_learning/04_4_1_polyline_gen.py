import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    x = numpy.linspace(0, 10, 100)
    y = numpy.piecewise(x, [x < 3, (x >= 3) & (x < 7), x >= 7],
                        [lambda x: x + numpy.random.normal(0, 0.5, len(x)),
                         lambda x: -0.5 * x + 7 + numpy.random.normal(0, 0.5, len(x)),
                         lambda x: x - 7 + numpy.random.normal(0, 0.5, len(x))])
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
