import numpy
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    LEARN_RATE = 0.01       # 学习率
    EPOCH = 4               # 训练轮数
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.random(size=100) - 0.5
    x_input_array = numpy.linspace(0, 4, 100)
    y_true_array = 3 * x_input_array + 4 + random_numbers
    param = 1.0             # 模型参数
    history = []

    for e in range(EPOCH):
        for index, x_input in enumerate(x_input_array):
            y_pred = param * x_input + 4
            if y_pred < y_true_array[index]:
                param += LEARN_RATE
            else:
                param -= LEARN_RATE
            history.append(param)

    pyplot.plot([0, EPOCH], [3, 3], linestyle='--')
    pyplot.plot(numpy.linspace(0, EPOCH, len(history)), history)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
