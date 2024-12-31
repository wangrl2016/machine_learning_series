import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.standard_normal(size=100)
    input = numpy.linspace(0, 4, 100)
    output_true = numpy.round(3 * input + 4 + random_numbers, 4)

    param = 1
    step = 0.01
    intermediate_list = [1.0]
    for index, x in enumerate(input):
        y_pred = numpy.round(param * x + 4, 4)
        if y_pred < output_true[index]:
            param += step
        else:
            param -= step
        intermediate_list.append(param)
    
    pyplot.plot(range(len(intermediate_list)), intermediate_list)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
