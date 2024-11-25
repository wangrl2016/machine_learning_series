import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.random(size=100) - 0.5
    x = numpy.linspace(0, 4, 100)
    y_true_array = 3 * x + 4 + random_numbers
    pyplot.scatter(x, y_true_array, s=5)
    pyplot.plot(x, x + 4, c='red')
    pyplot.text(3, 6, 'f(x) = x + 4')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

    step = 0.1
    param = 1.0
    y_pred = param * x[0] + 4
    if (y_pred < y_true_array[0]):
        param += step
    else:
        param -= step
    print(param)
    pyplot.scatter(x, y_true_array, s=5)
    pyplot.plot(x, x + 4, c='red', linestyle='--')
    pyplot.plot(x, param * x + 4, c='red')
    pyplot.text(3, 6, 'f(x) = x + 4')
    pyplot.text(2, 8, f'f(x) = {param} * x + 4')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
