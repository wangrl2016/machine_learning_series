from matplotlib import pyplot, animation
import numpy

def func(x):
    return 0.1*x**4 - numpy.sin(2.5*x) + 0.5*numpy.cos(1.5*x) + 0.2*x**2

def grad_func(x):
    return 0.4*x**3 - 2.5*numpy.cos(2.5*x) - 0.75*numpy.sin(1.5*x) + 0.4*x

# 小球初始位置
x_start = -3.0
# 学习率
learning_rate = 0.01
# 动量系数
momentum = 0.85
# 迭代次数
epochs = 100

if __name__ == '__main__':
    # 记录小球的位置
    x_values = [x_start]
    # 初始动量
    velocity = 0
    x = numpy.linspace(-3, 3, 600)

    for _ in range(epochs):
        grad = grad_func(x_values[-1])
        # 更新动量
        velocity = momentum * velocity - learning_rate * grad
        # 更新位置
        x_values.append(x_values[-1] + velocity)

    fig, ax = pyplot.subplots()
    ball, = ax.plot([], [], 'ro', markersize=10, label='Ball')
    path = ax.plot(x, func(x), linewidth=1, label='Path')
    ax.legend()
    ax.grid(True)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)

    def animate(i):
        ball.set_data([x_values[i]], [func(x_values[i])])
        return ball,

    anim = animation.FuncAnimation(fig, animate, frames=len(x_values), interval=40, blit=True, repeat=False)
    anim.save('temp/sgd_momentum.gif', writer='pillow', dpi=100)
    pyplot.show()
