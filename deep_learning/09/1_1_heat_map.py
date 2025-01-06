from matplotlib import pyplot
import numpy

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
    attentino_weights = numpy.eye(10).reshape((1, 1, 10, 10))
    print(attentino_weights.shape)
    show_heatmaps(attentino_weights)
