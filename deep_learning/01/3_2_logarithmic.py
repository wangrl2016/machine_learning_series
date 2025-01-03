import numpy
from matplotlib import pyplot

START = 0
END = 10
NUM = 1000
STEP = (END - START) / NUM

if __name__ == '__main__':
    x = numpy.arange(START + STEP, END + STEP, STEP)
    pyplot.plot(x, 2 * numpy.log(x) / numpy.log(4), label='f(x) = 2 * log4(x)')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
