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
