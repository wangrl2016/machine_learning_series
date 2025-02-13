from matplotlib import pyplot

def show_heatmap(matrices, x_label = '', y_label = '', cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = pyplot.subplots(num_rows, num_cols,
                                sharex=True, sharey=True, squeeze=False)
    # shape os axes: (1 x 1)
    for row_idx, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for col_idx, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix, cmap=cmap)
            if row_idx == num_rows - 1:
                ax.set_xlabel(x_label)
            if col_idx == 0:
                ax.set_ylabel(y_label)
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.1, wspace=0.2, hspace=0.2)
    pyplot.show()
