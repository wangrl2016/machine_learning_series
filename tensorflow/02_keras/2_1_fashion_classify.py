import keras
from matplotlib import pyplot
import numpy
import tensorflow as tf

if __name__ == '__main__':
    print('Keras version:', keras.__version__)
    print('TensorFlow version:', tf.__version__)

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

