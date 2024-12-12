import numpy

if __name__ == '__main__':
    a1 = numpy.array([1, 2])
    a2 = numpy.array([1, 1])
    a3 = a1 + a2
    assert (a3 == [2, 3]).all()
    a4 = a1 - a2
    assert (a4 == [0, 1]).all()
    a5 = a1 * a1
    assert (a5 == [1, 4]).all()
    a6 = a1 / a1
    assert numpy.allclose(a6, [1.0, 1.0])

    a7 = numpy.array([1, 2, 3, 4])
    assert a7.sum() == 10

    a8 = numpy.array([[1, 1], [2, 2]])
    assert (a8.sum(axis=0) == [3, 3]).all()
    assert (a8.sum(axis=1) == [2, 4]).all()
    
    a9 = numpy.array([1.0, 2.0])
    assert numpy.allclose(a9 * 1.6, [1.6, 3.2])
