import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x = numpy.linspace(0, 10)
    b_list = [-4, -2, 0, + 2, + 4]
    text_list = ['f(x) = x - 4', 'f(x) = x - 2', 'f(x) = x', 'f(x) = x + 2', 'f(x) = x + 4']
    for index, b in enumerate(b_list):
        y = x + b
        pyplot.text(5, 5 + b, text_list[index], fontsize=12)
        pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

    x = numpy.linspace(-4, 4)
    m_list = [-2, -1, -1/2,  1/2, 1, 2]
    text_list = ['f(x) = -2 * x', 'f(x) = -x', 'f(x) = -1/2 * x',
                 'f(x) = 1/2 * x', 'f(x) = x', 'f(x) = 2 * x']
    for index, m in enumerate(m_list):
        y = m * x
        pyplot.text(3, m * 3, text_list[index], fontsize=12)
        pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
