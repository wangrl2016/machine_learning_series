import numpy

if __name__ == '__main__':
    data = numpy.array([10, 20, 30, 40, 50])
    mean = data.mean()
    std = data.std()
    normalized = numpy.round((data - mean) / std, 2)
    print('Mean', mean)
    print('Std', std)
    print('Standardization', normalized)
