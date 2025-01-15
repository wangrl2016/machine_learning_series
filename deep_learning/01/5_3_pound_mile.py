from matplotlib import pyplot

if __name__ == '__main__':
    pounds = [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]
    miles = [18, 15, 18, 16, 15, 14, 24]
    
    pyplot.scatter(pounds, miles)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
