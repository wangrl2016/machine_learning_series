import numpy

if __name__ == '__main__':
    a1 = numpy.arange(10)
    assert (a1 == numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).all()
    a2 = numpy.arange(2, 10, step=2, dtype=float)
    assert numpy.allclose(a2, numpy.array([2.0, 4.0, 6.0, 8.0]))
    a3 = numpy.arange(2, 3, step=0.5)
    assert numpy.allclose(a3, numpy.array([2.0, 2.5]))

    a4 = numpy.linspace(4, 10, num=5)
    assert numpy.allclose(a4, numpy.array([4.0, 5.5, 7, 8.5, 10.0]))
    a5 = numpy.linspace(1.0, 4.0, 4, endpoint=False)
    assert numpy.allclose(a5, numpy.array([1.0, 1.75, 2.5, 3.25]))

    a6 = numpy.eye(3)
    print(a6)
    a7 = numpy.eye(3, 5)
    print(a7)
    a8 = numpy.diag([1, 2, 3])
    print(a8)
    a9 = numpy.diag([1, 2, 3], 1)
    print(a9)
    a10 = numpy.diag([[1, 2], [3, 4]])
    assert (a10 == numpy.array([1, 4])).all()

    a11 = numpy.vander(numpy.linspace(0, 2, 5), 2)
    print(a11)
    a12 = numpy.vander([1, 2, 3, 4], 3)
    print(a12)
    a13 = numpy.vander((1, 2, 3, 4), 4)
    print(a13)

    a14 = numpy.zeros((2, 3))
    print(a14)
    a15 = numpy.zeros((2, 3, 2))
    print(a15)

    a16 = numpy.ones((2, 3))
    print(a16)
    a17 = numpy.ones((2, 3, 2))
    print(a17)

    a18 = numpy.indices((2, 3))
    assert a18.shape == (2, 2, 3)
    print(a18)
    a19 = numpy.indices((2, 3, 4))
    print(a19)
