import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.standard_normal(size=100)
    print(random_numbers)
    x = numpy.linspace(0, 4, 100)
    y = 3 * x + 4 + random_numbers
    pyplot.scatter(x, y, s=5)
    pyplot.plot(x, 3 * x + 4, c='red',label='f(x) = 3x + 4')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
