import numpy

if __name__ == '__main__':
    data = numpy.array([10, 20, 30, 40, 50])
    result = numpy.zeros((5,))
    max = data.max()
    min = data.min()
    normalized = (data - min) / (max - min)
    print(normalized)
