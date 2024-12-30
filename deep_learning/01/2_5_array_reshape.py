import numpy

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    a1 = numpy.floor(10 * rng.random((3, 4)))
    assert a1.shape == (3, 4)

    # return the array, flattened
    a2 = a1.ravel()
    assert a2.shape == (12,)
    
    # returns the array with a modified shape
    a3 = a1.reshape(6, 2)
    assert a3.shape == (6, 2)
    
    # returns the array, transposed
    a4 = a1.T
    assert a4.shape == (4, 3)
    