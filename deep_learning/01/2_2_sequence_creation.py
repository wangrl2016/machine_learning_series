import numpy

if __name__ == '__main__':
    # 使用列表创建 ndarray
    list_data = [1, 2, 3, 4]
    a1 = numpy.array(list_data)
    a2 = numpy.array([[1, 2], [3, 4]])
    a3 = numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # 使用元组创建 ndarray
    tuple_data = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    a4 = numpy.array(tuple_data)
    
    # error
    # numpy.array([127, 128, 129], dtype=numpy.int8)

    a5 = numpy.array([2, 3, 4], dtype=numpy.uint32)
    a6 = numpy.array([5, 6, 7], dtype=numpy.uint32)
    assert ((a5 - a6) == numpy.array([4294967293, 4294967293, 4294967293])).all()
    assert ((a5 - a6.astype(numpy.int32)) == numpy.array([-3, -3, -3])).all()

    a7 = numpy.arange(10)
    assert (a7 == numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).all()
    a8 = numpy.arange(2, 10, dtype=float)
    assert numpy.allclose(a8, numpy.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
    a9 = numpy.arange(2, 3, 0.1)
    assert numpy.allclose(a9, numpy.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]))
    
    a10 = numpy.linspace(1.0, 4.0, 6)
    assert numpy.allclose(a10, numpy.array([1.0, 1.6, 2.2, 2.8, 3.4, 4.0]))
