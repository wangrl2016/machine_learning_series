from matplotlib import pyplot

if __name__ == '__main__':
    stock = [100, 102, 105, 101, 109, 114, 108, 100, 105, 108]
    pyplot.plot(range(len(stock)), stock, color='b')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
