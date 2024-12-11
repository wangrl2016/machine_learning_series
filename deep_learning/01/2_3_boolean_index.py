import numpy

if __name__ == '__main__':
    a1 = numpy.array([[1, 2], [numpy.nan, 3], [numpy.nan, numpy.nan]])
    a2 = a1[~numpy.isnan(a1)]
    assert (a2 == [1, 2, 3]).all()

    a3 = numpy.array([1, -1, -2, 3])
    a3[a3 < 0] += 20
    assert (a3 == [1, 19, 18, 3]).all() 

    a4 = numpy.arange(35).reshape(5, 7)
    a5 = a4 > 20
    assert (a5[:, 5] == [False, False, False, True, True]).all()
    a6 = a4[a5[:, 5]]
    assert (a6 == [[21, 22, 23, 24, 25, 26, 27],
                   [28, 29, 30, 31, 32, 33, 34]]).all()

    a7 = numpy.array([[0, 1], [1, 1], [2, 2]])
    row_sum = a7.sum(-1)
    assert (row_sum == [1, 2, 4]).all()
    a8 = a7[row_sum <= 2, :]
    assert (a8 == [[0, 1], [1, 1]]).all()

    a10 = numpy.array([[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8],
                       [9, 10, 11]])
    rows = (a10.sum(-1) % 2) == 0
    assert (rows == [False, True, False, True]).all()
    columns = [0, 2]
    a11 = a10[numpy.ix_(rows, columns)]
    assert (a11 == [[3, 5], [9, 11]]).all()

    rows = rows.nonzero()[0]
    assert (rows == numpy.array([1, 3])).all()
    a12 = a10[rows[:, numpy.newaxis], columns]
    assert (a12 == [[3, 5], [9, 11]]).all()
    
    a12 = numpy.arange(30).reshape(2, 3, 5)
    a13 = numpy.array([[True, True, False], [False, True, True]])
    a14 = a12[a13]
    assert (a14 == [[0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29]]).all()
    
    
