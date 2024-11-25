import numpy
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    LEARN_RATE = 0.01       # 学习率
    EPOCHS = 4               # 训练轮数
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.random(size=100) - 0.5
    x_input_array = numpy.linspace(0, 4, 100)
    y_true_array = 3 * x_input_array + 4 + random_numbers
    param = 1.0             # 模型参数
    history = []

    for _ in range(EPOCHS):
        for index, x_input in enumerate(x_input_array):
            y_pred = param * x_input + 4
            if y_pred < y_true_array[index]:
                param += LEARN_RATE
            else:
                param -= LEARN_RATE
            history.append(param)

    # history = history[::4]
    fig, ax = pyplot.subplots()
    line, = ax.plot(x_input_array, x_input_array + 4)
    text = ax.text(3, 10, 'param: 3.00')

    ax.scatter(x_input_array, y_true_array, s=5, c='blue')
    ax.plot(x_input_array, x_input_array + 4, c='red', linestyle='--', label='f(x) = x + 4')
    ax.legend()
    ax.grid(True)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)

    def animate(i):
        line.set_data(x_input_array, history[i] * x_input_array + 4)
        text.set_text(f'param: {history[i]:.2f}')
        return line, text

    animation = FuncAnimation(fig, animate, frames=len(history), interval=40, blit=True, repeat=False)
    animation.save("temp/simplest_ml_anim.gif", writer="pillow")
    pyplot.show()
