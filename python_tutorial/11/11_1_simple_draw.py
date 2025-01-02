from matplotlib import pyplot

if __name__ == '__main__':
    # Create a figure containing a single Axes.
    fig, ax = pyplot.subplots()
    # Plot some data on the Axes.
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    # Show the figure.
    pyplot.show()
