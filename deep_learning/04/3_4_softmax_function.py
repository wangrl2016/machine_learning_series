import numpy
from matplotlib import pyplot

def softmax(arr):
    sum = 0
    result = []
    for a in arr:
        sum += numpy.power(numpy.e, a)
    for a in arr:
        result.append(numpy.power(numpy.e, a) / sum)
    return result

if __name__ == '__main__':
    points = [2.0, 1.0, 0.1]
    e_list = []
    for point in points:
        data = numpy.power(numpy.e, point)
        e_list.append(float(f'{data:.3f}'))
        pyplot.scatter([point], [data], label=f'({point}, {data:.3f})')
    sum = numpy.sum(numpy.array(e_list))
    e_list = e_list / sum
    print('Method 1 probability', numpy.round(e_list, 4))
    print('Method 2 probability', numpy.round(softmax(points), 4))
    x = numpy.linspace(0, 2.5, 250)
    pyplot.plot(x, numpy.power(numpy.e, x), label='y = e^x')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
