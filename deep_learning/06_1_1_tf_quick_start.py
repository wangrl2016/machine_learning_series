import keras
import tensorflow

if __name__ == '__main__':
    print('TensorFlow version:', tensorflow.__version__)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    
    # 第 0 维进行切片操作
    predictions = model(x_train[:1]).numpy()
    print(predictions)
    probabilities = tensorflow.nn.softmax(predictions).numpy()
    print(probabilities)
    
    # 定义损失函数
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(y_train[:1], predictions).numpy()
    print(loss)

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)
    
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    print(probability_model(x_test[:3]))
