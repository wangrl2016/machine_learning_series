import keras
import numpy

if __name__ == '__main__':
    rng = numpy.random.default_rng(seed=0)
    input = rng.standard_normal((200, 2))
    output = numpy.array([1 if x + y > 0 else 0 for x, y in input])
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2,)))
    model.add(keras.layers.Dense(units=1, activation='sigmoid',
                                 kernel_initializer=keras.initializers.Constant(value=0),
                                 bias_initializer=keras.initializers.Constant(value=1.0)))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit(input, output, epochs=5, batch_size=1, verbose=1)
