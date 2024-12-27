import numpy

if __name__ == '__main__':
    rng = numpy.random.default_rng()
    embedding_vector = numpy.round(rng.random((6, 7)), 4)
    print(embedding_vector)
