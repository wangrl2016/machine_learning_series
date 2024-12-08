import numpy

if __name__ == '__main__':
    a1 = numpy.arange(10, 1, -1)
    assert (a1[numpy.array([3, 3, 1, 8])] == [7, 7, 9, 2]).all()
    assert (a1[numpy.array([3, 3, -3, 8])] == [7, 7, 4, 2]).all()

    a2 = numpy.array([[1, 2], [3, 4], [5, 6]])
    assert (a2[numpy.array([1, -1])] == [[3, 4], [5, 6]]).all()
    
    
