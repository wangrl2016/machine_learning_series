import numpy

word_list = ['When', 'you', 'play', 'the', 'game', 'of', 'thrones']

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    input_embedding = numpy.round(rng.random((6, 7)), 2)

    position_encoding = numpy.zeros((6, 7))
    for pos, word in enumerate(word_list):
        for i in range(6):
            if i % 2 == 0:
                position_encoding[i, pos] = \
                    numpy.round(numpy.sin(pos/(10000**(2*i/6))), 2)
            else:
                position_encoding[i, pos] = \
                    numpy.round(numpy.cos(pos/(10000**(2*i/6))), 2)
    
    result = input_embedding + position_encoding
    print(result)

    query = numpy.round(result.T @ rng.random((6, 4)), 2)
    key = numpy.round(result.T @ rng.random((6, 4)), 2)
    value = numpy.round(result.T @ rng.random((6, 4)), 2)    
    print(query)
    print(key)
    print(value)

    query_mul_key = numpy.round(query @ key.T, 2)
    print(query_mul_key)

    scale_query_mul_key = numpy.round(query_mul_key / 6**0.5, 2)
    print(scale_query_mul_key)

    softmax_scale = []
    
