from matplotlib import pyplot

if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    pyplot.scatter(x, y, marker='o')
    # 标记每个点的坐标
    for i in range(len(x)):
        pyplot.annotate(f'({x[i]}, {y[i]})', (x[i], y[i]),
                        textcoords='offset points', xytext=(5, 5), ha='center')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
