import numpy

if __name__ == '__main__':
    rng = numpy.random.default_rng()
    # Generate one random float uniformly distributed over the range [0, 1).
    print(rng.random()) # may vary
    # Generate an array of 10 numbers according to a unit Gaussian distribution.
    print(rng.standard_normal(10))
    # Generate an array of 5 integers uniformly over the range [0, 10).
    print(rng.integers(low=0, high=10, size=5))
    
    rng = numpy.random.default_rng(seed=0)
    print(rng.random())
