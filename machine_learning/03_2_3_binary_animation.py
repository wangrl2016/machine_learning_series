import keras
import numpy
from keras import initializers
from keras import optimizers
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

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
    model.fit(input, output, epochs=5, batch_size=1)
    
    fig, ax = pyplot.subplots()
    line, = pyplot.plot([-3, 3], [1, 1])
    weights_text = pyplot.text(1, 1, 'weights: []')
    bias_text = pyplot.text(-1, 1, 'bias:')

    pyplot.scatter(input[:, 0], input[:, 1], c=output, cmap='bwr', alpha=0.3)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    
    def animate(i):
        return line, weights_text, bias_text

    animation = FuncAnimation(fig, animate, frames=240, interval=40, blit=True, repeat=False)
    animation.save("temp/binary_classify_animation.gif", writer="pillow")
    pyplot.show()
    

