import keras
import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    input = rng.standard_normal((200, 2))
    output = numpy.array([1 if x + y > 0 else 0 for x, y in input])
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2,)))
    model.add(keras.layers.Dense(1, activation='sigmoid',
                                 kernel_initializer=keras.initializers.Constant(value=0),
                                 bias_initializer=keras.initializers.Constant(value=1.0)))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit(input, output, epochs=5, batch_size=1, verbose=1)

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        m = (-weights[0][0] / weights[0][1]).item()
        b = (-weights[1][0] / weights[0][1]).item()
        x = numpy.linspace(-3, 3)
        pyplot.plot(x, m * x + b)
        pyplot.text(2, m * 2 + b, f'y = {m:.2f}x + {b:.2f}')
    pyplot.scatter(input[:, 0], input[:, 1], c=output, cmap='bwr')
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
