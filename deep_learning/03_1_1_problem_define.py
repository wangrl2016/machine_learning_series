import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    input = rng.standard_normal((200, 2))
    output = numpy.array([1 if x + y > 0 else 0 for x, y in input])
    pyplot.scatter(input[:, 0], input[:, 1], c=output, cmap='bwr')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
