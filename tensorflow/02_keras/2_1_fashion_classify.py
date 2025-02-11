import keras
from matplotlib import pyplot
import numpy
import tensorflow as tf

if __name__ == '__main__':
    print('Keras version:', keras.__version__)
    print('TensorFlow version:', tf.__version__)

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(train_images.shape)
    print(len(train_labels))
    print(train_labels[:10])
    print(test_images.shape)
    print(len(test_labels))


    pyplot.imshow(train_images[0])
    pyplot.colorbar()
    pyplot.grid(False)
    pyplot.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06)
    pyplot.show()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.grid(False)
        pyplot.imshow(train_images[i], cmap=pyplot.cm.binary)
        pyplot.xlabel(class_names[train_labels[i]])
    pyplot.subplots_adjust(wspace=0.5, hspace=0.5,
                           left=0.04, right=0.96, top=0.96, bottom=0.06)
    pyplot.show()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy:', test_acc)

    probability_model = keras.Sequential([model,
                                          keras.layers.Softmax()])
    predictions = probability_model.predict(test_images[:5])
    print(numpy.argmax(predictions[0]))
    print(test_labels[0])

    # Grab an image from the test dataset.
    img = test_images[1]
    print(img.shape)
    # Add the image to a batch where it's the only member.
    img = numpy.expand_dims(img, axis=0)
    print(img.shape)

    pred = probability_model.predict(img)
    print(pred)
    print(numpy.argmax(pred[0]))
