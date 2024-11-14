import keras
from matplotlib import pyplot
import numpy
import tensorflow

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if __name__ == '__main__':
    print(tensorflow.__version__)
    (train_images, train_labels), (test_images, test_labels) = \
        keras.datasets.fashion_mnist.load_data()
    print(train_images.shape, test_images.shape)
    
    # show first image
    pyplot.imshow(train_images[0])
    pyplot.colorbar()
    pyplot.show()
    pyplot.close()

    train_images, test_images = train_images / 255.0, test_images / 255.0
    for i in range(20):
        pyplot.subplot(4, 5, i + 1)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(train_images[i])
        pyplot.xlabel(CLASS_NAMES[train_labels[i]])
    pyplot.subplots_adjust(top=0.95, bottom=0.05)
    pyplot.show()

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy:', test_acc)
    
    # make predictions
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print(predictions[0])
    print(numpy.argmax(predictions[0]), test_labels[0])
