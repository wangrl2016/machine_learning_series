import numpy
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    LEARN_RATE = 0.01       # 学习率
    EPOCHS = 4               # 训练轮数
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.random(size=100) - 0.5
    input_array = numpy.linspace(0, 4, 100)
    output_array = 3 * input_array + 4 + random_numbers
    param = 1.0             # 模型参数
    history = []

    for _ in range(EPOCHS):
        for input, output in zip(input_array, output_array):
            predict = param * input + 4
            if predict < output:
                param += LEARN_RATE
            else:
                param -= LEARN_RATE
            history.append(param)

    pyplot.plot([0, EPOCHS], [3, 3], linestyle='--')
    pyplot.plot(numpy.linspace(0, EPOCHS, len(history)), history)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
