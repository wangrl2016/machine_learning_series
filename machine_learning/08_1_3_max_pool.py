import numpy

def max_pooling(input_matrix, pool_size, stride):
    rows = (input_matrix.shape[0] - pool_size) // stride + 1
    cols = (input_matrix.shape[1] - pool_size) // stride + 1
    output = numpy.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            start_row, start_col = i * stride, j * stride
            end_row, end_col = start_row + pool_size, start_col + pool_size
            output[i, j] = numpy.max(input_matrix[start_row:end_row, start_col:end_col])
    return output

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    input = numpy.array([
        [1, 3, 2, 4],
        [5, 6, 8, 9],
        [3, 2, 1, 0],
        [7, 2, 4, 6]
    ])

    output = max_pooling(input, pool_size=2, stride=2)
    print(output)
