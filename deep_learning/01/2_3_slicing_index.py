import numpy

if __name__ == '__main__':
    a1 = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert (a1[1:7:2] == [1, 3, 5]).all()

    assert (a1[-2:10] == [8, 9]).all()
    assert (a1[-3:3:-1] == [7, 6, 5, 4]).all()

    assert (a1[5:] == [5, 6, 7, 8, 9]).all()

    a2 = numpy.array([[[1],[2],[3]], [[4],[5],[6]]])
    assert (a2[1:2] == [[[4], [5], [6]]]).all()

    a3 = numpy.arange(27).reshape(3, 3, 3)
    assert a3[0].shape == (3, 3)
    assert a3[0:1].shape == (1, 3, 3)
    a4 = numpy.array([100, 200, 300])
    assert a4[1] == 200

    a5 = numpy.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert (a5[:, 0:3:2, :] == [[[0, 1, 2, 3], [8, 9, 10, 11]],
                                 [[12, 13, 14, 15], [20, 21, 22, 23]]]).all()

    a6 = numpy.arange(3*4*5).reshape(3, 4, 5)
    assert (a6[1, 2, :] == [30, 31, 32, 33, 34]).all()
    assert (a6[1, 2, :] == a6[1][2][:]).all()

    a7 = numpy.zeros((3, 3))
    a7[1, :] = [10, 20, 30]
    assert numpy.allclose(a7, [[0, 0, 0], [10, 20, 30], [0, 0, 0]])

    a8 = numpy.arange(10*2).reshape(10, 2)
    obj = (slice(1, 10, 5), slice(None, None, -1))
    assert (a8[1:10:5, ::-1] == [[3, 2], [13, 12]]).all()
    assert (a8[1:10:5, ::-1] == a8[obj]).all()
