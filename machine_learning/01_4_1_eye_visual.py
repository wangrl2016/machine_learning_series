import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.random(size=100) - 0.5
    x = numpy.linspace(0, 4, 100)
    pyplot.scatter(x, 3 * x + 4 + random_numbers, s=5)
    pyplot.plot(x, 3 * x + 4, label='y = 3 * x + 4', color='red')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
