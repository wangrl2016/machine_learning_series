import numpy

if __name__ == '__main__':
    a1 = numpy.array([1, 2, 3, 4, 5, 6])
    a2 = a1[0:3]
    a2 += 2
    assert (a1 == [3, 4, 5, 4, 5, 6]).all()
    assert (a2 == [3, 4, 5]).all()
    print('a1 =', a1, '; a2 =', a2)
    
    a3 = numpy.array([1, 2, 3, 4, 5, 6])
    a4 = a3[0:3].copy()
    a4 += 2
    assert (a3 == [1, 2, 3, 4, 5, 6]).all()
    assert (a4 == [3, 4, 5]).all()
    print('a3 =',a3, '; a4 =', a4)

    a5 = numpy.array([[1, 2, 3], [4, 5, 6]])
    a6 = numpy.array([[7, 8, 9], [10, 11, 12]])
    a7 = numpy.vstack((a5, a6))
    a8 = numpy.hstack((a5, a6))
    print(a7)
    print(a8)
    
    a9 = numpy.ones((2, 2))
    a10 = numpy.eye(2, 2)
    a11 = numpy.zeros((2, 2))
    a12 = numpy.diag((3, 4))
    a13 = numpy.block([[a9, a10], [a11, a12]])
    print(a13)
