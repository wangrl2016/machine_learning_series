import numpy

if __name__ == '__main__':
    a1 = numpy.arange(10, 1, -1)
    assert (a1[numpy.array([3, 3, 1, 8])] == [7, 7, 9, 2]).all()
    assert (a1[numpy.array([3, 3, -3, 8])] == [7, 7, 4, 2]).all()
    
    a2 = numpy.array([[1, 2], [3, 4], [5, 6]])
    assert (a2[numpy.array([1, -1])] == [[3, 4], [5, 6]]).all()

    a3 = numpy.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    rows = numpy.array([0, 1, 2])
    cols = numpy.array([1, 2, 0])
    assert (a3[rows, cols] == [20, 60, 70]).all()

    rows = numpy.array([[0], [1], [2]])
    cols = numpy.array([0, 1, 2])
    assert (a3 == a3[rows, cols]).all()
    
    a4 = numpy.arange(27).reshape(3, 3, 3)
    dim1 = numpy.array([0, 1])
    dim2 = numpy.array([[0], [2]])
    dim3 = numpy.array([1, 2])
    a5 = a4[dim1, dim2, dim3]
    assert (a5 == [[1, 11], [7, 17]]).all()

    a6 = numpy.arange(35).reshape(5, 7)
    assert (a6[numpy.array([0, 2, 4]), numpy.array([0, 1, 2])] == [0, 15, 30]).all()
    # IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,) (2,)
    # a6[numpy.array([0, 2, 4]), numpy.array([0, 1])]
    assert (a6[numpy.array([0, 2, 4]), 1] == [1, 15, 29]).all()
    
    a7 = a6[numpy.array([0, 2, 4])]
    assert(a7 == [[0, 1, 2, 3, 4, 5, 6],
                  [14, 15, 16, 17, 18, 19, 20],
                  [28, 29, 30, 31, 32, 33, 34]]).all()
    