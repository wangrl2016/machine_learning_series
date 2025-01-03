import keras
import numpy

if __name__ == '__main__':
    # 真实标签
    y_true = numpy.array([1.0, 0.0, 1.0, 1.0])
    # 预测值
    y_pred = numpy.array([0.9, 0.1, 0.8, 0.7])

    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    print(loss(y_true, y_pred).numpy())
