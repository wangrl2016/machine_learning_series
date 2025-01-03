import keras
import numpy

if __name__ == '__main__':
    y_true = numpy.array([[0, 1, 0], [0, 0, 1]])
    y_pred = numpy.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    cce = keras.losses.CategoricalCrossentropy(from_logits=False)
    print(cce(y_true, y_pred).numpy())
