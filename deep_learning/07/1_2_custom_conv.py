import numpy
import scipy
import scipy.ndimage

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    input = rng.integers(0, 100, (5, 5))
    kernel = rng.integers(0, 2, (3, 3))
    print(input)
    print(kernel)

    # scipy 进行卷积操作
    print('Result of scipy convolution:')
    result1 = scipy.ndimage.convolve(input, kernel, mode='constant', cval=0)
    print(result1)

    # 使用 numpy.pad 在边缘填充 0
    input_padded = numpy.pad(input, pad_width=1, mode='constant', constant_values=0)
    print(input_padded)
    # 翻转卷积核
    kernel_flipped = numpy.flip(kernel)
    print(kernel_flipped)
    input_height, input_width = input_padded.shape
    kernel_height, kernel_width = kernel.shape
    result2 = numpy.zeros((input_height - 2, input_width - 2), dtype=int)
    for i in range(1, input_height - 1):
        for j in range(1, input_width - 1):
            # 提取当前 3x3 的子区域
            sub_matrix = input_padded[i - 1:i + 2, j - 1:j + 2]
            # 计算卷积：对 sub_matrix 和 kernel 的对应元素相乘并求和
            result2[i - 1, j - 1] = numpy.sum(sub_matrix * kernel_flipped)
    assert(result1 == result2).all()
    print('Result of custom convolution:')
    print(result2)
