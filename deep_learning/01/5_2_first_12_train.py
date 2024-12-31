import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.standard_normal(size=100)
    input = numpy.linspace(0, 4, 100)
    output_true = numpy.round(3 * input + 4 + random_numbers, 4)

    param = 1
    step = 0.01
    for index, x in enumerate(input):
        y_pred = numpy.round(param * x + 4, 4)
        change = ''
        if y_pred < output_true[index]:
            param += step
            change = '+' + str(step)
        else:
            param -= step
            change = '-' + str(step)
        print(index, numpy.round(input[index], 4), y_pred, output_true[index], change, param)
        if index > 10:
            break
