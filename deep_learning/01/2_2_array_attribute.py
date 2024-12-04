import numpy

if __name__ == '__main__':
    a1 = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert a1.ndim == 2
    assert a1.shape == (3, 4)
    assert a1.size == 12
    assert a1.dtype == 'int64'
