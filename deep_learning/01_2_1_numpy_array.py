import numpy

if __name__ == '__main__':
    a = numpy.array([1, 2, 3, 4, 5, 6])
    print(a)
    print(numpy.array([[1, 2, 3, 4, 5],
                       [5, 7, 8, 9, 10],
                       [11, 12, 13, 14, 15]]))

    b = numpy.array([[1, 2], [3, 4]])
    c = numpy.array([[5, 6]])
    print(numpy.concatenate((b, c), axis=0))

    d = numpy.array([1, 2, 3, 4, 5, 6])
    e = d.reshape(3, 2)
    print(e.shape)

    f = numpy.array([1, 2, 3, 4, 5, 6])
    g = f[numpy.newaxis, :]
    print(g.shape)

    h = numpy.array([1, 2, 3])
    print(h[1], h[0:2], h[1:], h[-2:])

    data = numpy.array([1, 2])
    ones = numpy.ones(2, dtype=int)
    print(data + ones)

    data = numpy.array([1.0, 2.0])
    print(data * 1.6)

    data = numpy.array([1.0, 2.0, 3.0])
    print(data.max(), data.min(), data.sum())
