
import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.random(size=100) - 0.5
    x = numpy.linspace(0, 4, 100)
    y = 3 * x + 4 + random_numbers
    # 输入前面两个点的坐标
    for i in range(2):
        print('(', x[i], ',', y[i], ')')

    sum_dict = {}
    for m in numpy.linspace(2, 4, 200):
        distance_square_sum = 0
        for i in range(len(x)):
            distance_square_sum += numpy.power(m * x[i] - y[i] + 4, 2) / (m * m + 1)
        sum_dict[m] = distance_square_sum
    min_key = min(sum_dict, key=sum_dict.get)
    print(min_key)

    # 绘制距离平方和与斜率的关系图像
    pyplot.plot(sum_dict.keys(), sum_dict.values())
    pyplot.grid(True)
    pyplot.scatter([min_key], [sum_dict[min_key]], label=f'({min_key}, {sum_dict[min_key]}')
    pyplot.ylabel('sum(d^2)')
    pyplot.legend()
    pyplot.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.06)
    pyplot.show()
