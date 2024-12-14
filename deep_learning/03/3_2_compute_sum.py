import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras

class ComputeSum(keras.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = self.add_weight(
            shape=(),
            initializer='zeros',
            trainable=False,
            name='total'
        )
    def call(self, inputs):
        self.total.assign(self.total + keras.ops.sum(inputs))
        return self.total

if __name__ == '__main__':
    my_sum = ComputeSum(2)
    x = keras.ops.ones((2, 2))
    y = my_sum(x)
    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
