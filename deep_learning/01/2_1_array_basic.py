import numpy

if __name__ == '__main__':
    a1 = numpy.array([1, 2, 3, 4, 5, 6])
    print('Numpy array:', a1)
    print('First element:', a1[0])
    a1[0] = 10
    print('Change first element:', a1[0])

    a2 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print('Two dimensional array:', a2.tolist())
    print('Element in row 1 and column 3:', a2[1, 3])
    
    a3 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert a3.ndim == 2
    assert a3.shape == (3, 4)
    assert a3.size == 12
    assert a3.dtype == 'int64'
