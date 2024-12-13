import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras
import numpy

if __name__ == '__main__':
    layer = keras.layers.Dense(8, activation='relu')
    input = keras.random.uniform(shape=(3, 7))
    assert input.shape == (3, 7)
    output = layer(input)
    assert output.shape == (3, 8)

    print(layer.weights)
    numpy.set_printoptions(precision=4)
    for weight in layer.weights:
        print(weight.name, weight.numpy())
