import numpy
from matplotlib import pyplot
import pandas

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    random_numbers = rng.random(size=100) - 0.5
    x = numpy.linspace(0, 4, 100)
    pyplot.scatter(x, 3 * x + 4 + random_numbers, s=5)
    pyplot.plot(x, 3 * x + 4, c='red')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

    points_df = pandas.DataFrame({'X': x, 'Y': 3 * x + 4 + random_numbers})
    points_df['Coordinates'] = points_df.apply(lambda row: f"({row['X']:.6f}, {row['Y']:.6f})", axis=1)
    grid_data = points_df['Coordinates'].values.reshape(25, 4)
    fig, ax = pyplot.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=grid_data, cellLoc='center', loc='center')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    pyplot.show()
