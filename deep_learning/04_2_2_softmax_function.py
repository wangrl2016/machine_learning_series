import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    points = [2.0, 1.0, 0.1]
    e_list = []
    for point in points:
        data = numpy.power(numpy.e, point)
        e_list.append(float(f'{data:.3f}'))
        pyplot.scatter([point], [data], label=f'({point}, {data:.3f})')
    print('data list', e_list)
    sum = numpy.sum(numpy.array(e_list))
    print('sum', sum)
    e_list = e_list / sum
    print('probability', e_list)

    x = numpy.linspace(0, 2.5, 250)
    pyplot.plot(x, numpy.power(numpy.e, x), label='y = e^x')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
    
