import numpy

word_list = ['When', 'you', 'play', 'the', 'game', 'of', 'thrones']

if __name__ == '__main__':
    result = numpy.zeros((6, 7))

    for pos, word in enumerate(word_list):
        for i in range(6):
            if i % 2 == 0:
                result[i, pos] = numpy.round(numpy.sin(pos/numpy.power(10000, 2*i/6)), 4)
            else:
                result[i, pos] = numpy.round(numpy.cos(pos/numpy.power(10000, 2*i/6)), 4)
    print(result)
