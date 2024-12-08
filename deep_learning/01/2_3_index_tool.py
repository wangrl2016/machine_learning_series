import numpy

if __name__ == '__main__':
    a1 = numpy.arange(2*3*4*5).reshape(2, 3, 4, 5)
    a2 = a1[1, ..., 2]
    assert (a2 == a1[1, :, :, 2]).all()
    assert a2.shape == (3, 4)
    a3 = a1[..., 3]
    assert (a3 == a1[:, :, :, 3]).all()
    assert a3.shape == (2, 3, 4)

    a4 = numpy.arange(2*3*4).reshape(2, 3, 4)
    a5 = a4[:, numpy.newaxis, :, :]
    assert (a5.shape == (2, 1, 3, 4))
    a6 = numpy.arange(5)
    a7 = a6[:, numpy.newaxis] + a6[numpy.newaxis, :]
    print(a7)
