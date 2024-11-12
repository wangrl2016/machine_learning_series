import numpy

def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

if __name__ == '__main__':
    pass
