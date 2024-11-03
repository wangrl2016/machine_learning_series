from matplotlib import pyplot

if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    pyplot.scatter(x, y, marker='o')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
