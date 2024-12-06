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
    
    a5 = numpy.ones((2, 2))
    a6 = numpy.eye(2, 2)
    a7 = numpy.zeros((2, 2))
    a8 = numpy.diag((3, 4))
    a9 = numpy.block([[a5, a6], [a7, a8]])
    print(a9)
