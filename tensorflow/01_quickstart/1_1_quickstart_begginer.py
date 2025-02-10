import keras
import tensorflow as tf
import numpy

if __name__ == '__main__':
    print('Keras version:', keras.__version__)
    print('TensorFlow version:', tf.__version__)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10)
    ])

    NUM = 5
    predictions = model.call(x_train[:NUM]).numpy()
    print(numpy.round(predictions, 3))
    
    probabilities = tf.nn.softmax(predictions).numpy()
    print(numpy.round(probabilities, 3))

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print(loss_fn(y_train[:NUM], predictions).numpy())

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)

    probability_model = keras.Sequential([
        model,
        keras.layers.Softmax()
    ])

    probabilities = probability_model(x_test[:NUM]).numpy()
    print(numpy.round(probabilities, 3))
    print('Predict label:', probabilities.argmax(axis=-1))
    print('True label:', y_test[:NUM])
