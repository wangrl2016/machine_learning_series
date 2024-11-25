import numpy
from matplotlib import pyplot, animation, ticker

# Define the Rosenbrock function.
def rosenbrock(x, y, a=5, b=10):
    return (a-x)**2 + b * (y-x**2)**2

def gradient(x, y, a=5, b=10):
    dr_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dr_dy = 2 * b * (y - x**2)
    return dr_dx, dr_dy

LREARING_RATE = 0.002
STEP = 400

if __name__ == '__main__':
    # starting point
    x_vals, y_vals = [-1.5], [1]
    # Perform gradient descent and store each step.
    for _ in range(STEP):
        dx, dy = gradient(x_vals[-1], y_vals[-1])
        x_vals.append(x_vals[-1] - LREARING_RATE * dx)
        y_vals.append(y_vals[-1] - LREARING_RATE * dy)
    
    # Prepare the 3D surface plot.
    x = numpy.linspace(-2, 2, 400)
    y = numpy.linspace(-1, 3, 400)
    x, y = numpy.meshgrid(x, y)
    z = rosenbrock(x, y)

    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap='viridis')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.05)
    # 降低网格密度
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Initialize the point and annotation.
    point, = ax.plot([], [], [], 'ro', markersize=6)
    annotation = ax.text(-2, 0, 250, '', color='black')
    path_line, = ax.plot([], [], [], color='white', linestyle='-', linewidth=1)

    def animate(i):
        x, y = x_vals[i], y_vals[i]
        z = rosenbrock(x, y)
        point.set_data([x], [y])
        point.set_3d_properties([z])
        annotation.set_text(f'Step {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}')
        path_line.set_data(x_vals[:i+1], y_vals[:i+1])
        path_line.set_3d_properties([rosenbrock(x, y) for x, y in zip(x_vals[:i+1], y_vals[:i+1])])
        return point, annotation, path_line

    anim = animation.FuncAnimation(fig, animate,
                                   frames=min(STEP, len(x_vals)),
                                   blit=True, interval=40, repeat=False)
    # TODO: There is a problem with the saved line and point colors
    anim.save('temp/gradient_descent.gif', writer='pillow')
    pyplot.show()
