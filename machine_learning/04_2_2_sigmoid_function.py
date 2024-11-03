import numpy
from matplotlib import pyplot

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

if __name__ == '__main__':
    x = numpy.linspace(-10, 10, 400)
    y = sigmoid(x)
    pyplot.plot(x, y, label='1 / (1 + e^-x)')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.axvline(0, color='red', linewidth=2)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
