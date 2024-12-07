import numpy

if __name__ == '__main__':
    a1 = numpy.arange(10)
    assert a1[2] == 2
    assert a1[-2] == 8

    a1.shape = (2, 5)
    assert a1[1, 3] == 8
    assert a1[1, -1] == 9
    assert (a1[0] == [0, 1, 2, 3, 4]).all()
    assert a1[0, 2] == a1[0][2]
