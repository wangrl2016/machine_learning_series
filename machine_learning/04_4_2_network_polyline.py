import keras
import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    x = numpy.linspace(0, 10, 100)
    y = numpy.piecewise(x, [x < 3, (x >= 3) & (x < 7), x >= 7],
                        [lambda x: x + numpy.random.normal(0, 0.5, len(x)),
                         lambda x: -0.5 * x + 7 + numpy.random.normal(0, 0.5, len(x)),
                         lambda x: x - 7 + numpy.random.normal(0, 0.5, len(x))])
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    model = keras.Sequential([
        keras.layers.Dense(48, activation='relu', input_shape=(1,)),
        keras.layers.Dense(36, activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1)
    ])
    optimizer= keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(x, y, epochs=100)
    
    y_predict = model.predict(x)
    pyplot.plot(x, y)
    pyplot.scatter(x, y, color='green')
    pyplot.plot(x, y_predict, color='red', label='Fitting Curve')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
