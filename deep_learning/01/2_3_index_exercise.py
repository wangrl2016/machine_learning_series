import numpy

if __name__ == '__main__':
    arr = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    assert (arr[1, 2:4] == [7, 8]).all()
    assert (arr[:, 1] == [2, 6, 10, 14]).all()
    assert (arr[2:, 2:] == [[11, 12], [15, 16]]).all()
    assert (arr[1::2, ::2] == [[5, 7], [13, 15]]).all()
