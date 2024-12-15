import numpy
from matplotlib import pyplot, animation
from functools import partial

if __name__ == '__main__':
    fig, ax = pyplot.subplots()
    ax.set_xlim(-3, 5)
    ax.set_ylim(-2, 5)
    x = numpy.linspace(-2, 2, 400)
    pyplot.plot(x, x**2, label='y = x^2')
    ax.grid(True)
    ax.legend()
    fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)

    line, = pyplot.plot([], [])
    path, = pyplot.plot([], [], linestyle='--')
    path_data = []
    def animate(index):
        bound = 100
        step = 0.02
        if index < bound:
            offset = index * step
            x = numpy.linspace(-2 + offset, 2 + offset, 400)
            line.set_data(x, (x - offset)**2)
            path_data.append((offset, 0))
            path.set_data(*zip(*path_data))
        else:
            x = numpy.linspace(0, 4, 400)
            line.set_data(x, (x - bound * step)**2 - step * (index - bound))
            path_data.append((2, -step * (index - bound)))
            path.set_data(*zip(*path_data))
        return line, path

    anim = animation.FuncAnimation(fig, animate, frames=150,
                                   interval=40, blit=True, repeat=False)
    anim.save('temp/shift_anim.gif', writer='pillow', dpi=100)
    path_data = []
    pyplot.show()
