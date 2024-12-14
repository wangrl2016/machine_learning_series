import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras
import numpy
import seaborn
from matplotlib import pyplot
from matplotlib import colors

if __name__ == '__main__':
    layer = keras.layers.Dense(7, activation='relu')
    input = keras.random.uniform(shape=(3, 5))
    assert input.shape == (3, 5)
    output = layer(input)
    assert output.shape == (3, 7)

    print(layer.weights)
    numpy.set_printoptions(precision=4)
    for weight in layer.weights:
        print(weight.name, weight.numpy())

    assert len(layer.weights) == 2
    w = layer.weights[0]
    b = layer.weights[1]
    result = numpy.maximum(0, input @ w + b)
    assert numpy.allclose(output, result)

    seaborn.heatmap(result, annot=True, fmt='.4f',
                    cmap=colors.LinearSegmentedColormap.from_list('single_color', ['#FFF2CC', '#FFF2CC']),
                    linewidths=1, linecolor='black', cbar=False)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.axis('off')
    pyplot.show()
