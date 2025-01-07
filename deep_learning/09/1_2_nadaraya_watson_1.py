import numpy
from matplotlib import pyplot

def func(x):
    return 2 * numpy.sin(x) + x**0.8

def softmax(x):
    exp_x = numpy.exp(x)
    return exp_x / numpy.sum(exp_x, axis=-1, keepdims=True)

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    n_train = 50
    x_train = numpy.sort(rng.random(n_train) * 5)
    y_train = func(x_train) + rng.normal(0.0, 0.5, (n_train,))
    x_test = numpy.arange(0, 5, 0.1)
    y_test = func(x_test)
    n_test = len(x_test)
    x_repeat = x_test.repeat(n_train).reshape((-1, n_train))
    print(x_repeat)
    attention_weights = softmax(-0.5*(x_repeat - x_train)**2)
    y_hat = numpy.dot(attention_weights, y_train)

    pyplot.plot(x_train, y_train, 'o', alpha=0.5, label='Samples')
    pyplot.plot(x_test, y_test, label='True')
    pyplot.plot(x_test, y_hat, '--', label='Pred')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
