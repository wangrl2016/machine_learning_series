import numpy

def binary_cross_entropy(y_pred, y_true):
    return -(y_true * numpy.log(y_pred) + (1 - y_true) * numpy.log(1 - y_pred))

def deriv_binary_cross_entry(y_pred, y_true):
    return y_pred - y_true


if __name__ == '__main__':
    y = 1
    p = 0.9
    # 0.10536051565782628
    print(binary_cross_entropy(p, y))

    y = 1
    p = 0.1
    # 2.3025850929940455
    print(binary_cross_entropy(p, y))
