import numpy

if __name__ == '__main__':
    a1 = numpy.array([1.0, 2.0])
    assert numpy.allclose(a1 * 1.6, [1.6, 3.2])
