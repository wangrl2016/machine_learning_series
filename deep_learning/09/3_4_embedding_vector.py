import numpy

if __name__ == '__main__':
    rng = numpy.random.default_rng()
    embedding_vector = numpy.round(rng.random((7, 6)), 4)
    print(embedding_vector)
