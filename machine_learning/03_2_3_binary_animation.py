import keras
import numpy
from keras import initializers
from keras import optimizers
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

class ParamsCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.weights_history =[]
        self.bias_history = []

    def on_train_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            weights, bias = layer.get_weights()
            self.weights_history.append(weights)
            self.bias_history.append(bias)

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    input = rng.standard_normal((200, 2))
    output = numpy.array([1 if x + y > 0 else 0 for x, y in input])
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2,)))
    model.add(keras.layers.Dense(units=1, activation='sigmoid',
                                 kernel_initializer=initializers.Constant(0.0),
                                 bias_initializer=initializers.Constant(1.0)))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    params_callback = ParamsCallback()
    model.fit(input, output, epochs=4, batch_size=2, callbacks=[params_callback])

    fig, ax = pyplot.subplots()
    line, = pyplot.plot([], [])
    weights_text = pyplot.text(-1, -3, 'weights: [0, 0]')
    bias_text = pyplot.text(1, -3, 'bias: 1')

    pyplot.scatter(input[:, 0], input[:, 1], c=output, cmap='bwr')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)

    def animate(i):
        weights = params_callback.weights_history[i]
        bias = params_callback.bias_history[i]
        x = [-3, 3]
        if weights[1] == 0:
            y = [bias[0], bias[0]]
        else:
            y = - weights[0] / weights[1] * x + bias / weights[1]
        line.set_data(x, y)
        weights_text.set_text(f'weights: [{weights[0][0]:.2f}, {weights[1][0]:.2f}]')
        bias_text.set_text(f'bias: {bias[0]:.2f}')
        return line, weights_text, bias_text

    animation = FuncAnimation(fig, animate, frames=len(params_callback.weights_history),
                              interval=40, blit=True, repeat=False)
    animation.save("temp/binary_classify_anim.gif", writer="pillow")
    pyplot.show()
