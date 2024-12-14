import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras

class SimpleDense(keras.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel',
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias',
        )
    
    def call(self, inputs):
        return keras.ops.matmul(inputs, self.kernel) + self.bias
    
if __name__ == '__main__':
    # Instantiates the layer.
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(keras.ops.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `trainable_weights`.
    assert len(linear_layer.trainable_weights) == 2
