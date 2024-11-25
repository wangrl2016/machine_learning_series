import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x_values = [0, 5]
    y_values = [250, 665]
    x = numpy.linspace(0, 5)
    y = 83 * x + 250
    for i, txt in enumerate(zip(x_values, y_values)):
        pyplot.annotate(f'{txt}', (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, 10))
    pyplot.text(2.5, 400, 'D(t) = 83t + 250', fontsize=12, color='blue')
    pyplot.scatter(x_values, y_values, color='red')
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

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
