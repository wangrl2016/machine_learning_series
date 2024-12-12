import numpy

if __name__ == '__main__':
    a1 = numpy.arange(35).reshape(5, 7)
    a2 = a1[numpy.array([0, 2, 4]), 1:3]
    assert (a2 == [[1, 2], [15, 16], [29, 30]]).all()

    a3 = a1[:, 1:3][numpy.array([0, 2, 4]), :]
    assert (a2 == a3).all()

    a4 = numpy.array([[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8],
                      [9, 10, 11]])
    assert (a4[1:2, 1:3] == [[4, 5]]).all()
    assert (a4[1:2, [1, 2]] == [[4, 5]]).all()

    a5 = numpy.arange(35).reshape(5, 7)
    a6 = a5 > 20
    a7 = a5[a6[:, 5], 1:3]
    assert (a7 == [[22, 23], [29, 30]]).all()
