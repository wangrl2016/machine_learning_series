from matplotlib import pyplot
import numpy

def func(x):
    return 2 * numpy.sin(x) + x**0.8

def softmax(x):
    exp_x = numpy.exp(x)
    return exp_x / numpy.sum(exp_x, axis=-1, keepdims=True)

def show_heatmaps(matrices, cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = pyplot.subplots(num_rows, num_cols,
                                sharex=True, sharey=True, squeeze=False)
    for (row_axes, row_matrices) in zip(axes, matrices):
        for (ax, matrix) in zip(row_axes, row_matrices):
            pcm = ax.imshow(matrix, cmap=cmap)
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    pyplot.show()

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    n_train = 50
    x_test = numpy.arange(0, 5, 0.1)
    x_train = numpy.sort(rng.random(n_train) * 5)
    y_train = func(x_train) + rng.normal(0.0, 0.5, (n_train,))
    x_repeat = x_test.repeat(n_train).reshape((-1, n_train))
    attention_weights = softmax(-0.5*(x_repeat - x_train)**2)
    print(attention_weights.shape)
    attention_weights = numpy.expand_dims(numpy.expand_dims(attention_weights, 0), 0)
    show_heatmaps(attention_weights)
