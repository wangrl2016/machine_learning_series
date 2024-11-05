import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng()
    data = rng.standard_normal(10000)
    pyplot.hist(data, bins=50, density=True)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
