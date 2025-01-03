import tensorflow
import deep_learning.dataset.mnist as mnist

LAYER_SIZES = [784, 256, 128, 10]
BATCH_SIZE = 32
EPOCHS = 5

if __name__ == '__main__':
    print('TensorFlow version:', tensorflow.__version__)
    weights = {
        'h1': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[0], LAYER_SIZES[1]])),
        'h2': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[1], LAYER_SIZES[2]])),
        'out': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[2], LAYER_SIZES[3]])),}
    biases = {
        'b1': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[1]])),
        'b2': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[2]])),
        'out': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[3]])),}
    
    # 定义网络结构
    def NeuralNetwork(x):
        layer_1 = tensorflow.add(tensorflow.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tensorflow.nn.relu(layer_1)
        layer_2 = tensorflow.add(tensorflow.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tensorflow.nn.relu(layer_2)
        out_layer = tensorflow.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    class CustomSGD:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

        def apply_gradients(self, grads_and_vars):
            for grad, var in grads_and_vars:
                for grad, var in grads_and_vars:
                    if grad is not None:
                        var.assign_sub(self.learning_rate * grad)

    loss_fn = tensorflow.nn.softmax_cross_entropy_with_logits
    optimizer = CustomSGD(learning_rate=0.001)

    def train_step(x, y):
        with tensorflow.GradientTape() as tape:
            logits = NeuralNetwork(x)
            loss = tensorflow.reduce_mean(loss_fn(labels=y, logits=logits))
        gradients = tape.gradient(loss, list(weights.values()) + list(biases.values()))
        optimizer.apply_gradients(zip(gradients, list(weights.values()) + list(biases.values())))
        return loss

    def test_step(x, y):
        logits = NeuralNetwork(x)
        correct_prediction = tensorflow.equal(tensorflow.argmax(logits, 1), tensorflow.argmax(y, 1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
        return accuracy

    (x_train, y_train), (x_test, y_test) = mnist.parse()
    x_train, x_test = x_train.reshape(-1, 784).astype('float32') / 255.0, x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train, y_test = tensorflow.one_hot(y_train, depth=10), tensorflow.one_hot(y_test, depth=10)
    train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    test_dataset = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        for x_batch, y_batch in train_dataset:
            loss = train_step(x_batch, y_batch)
        test_accuracy = tensorflow.reduce_mean([test_step(x, y) for x, y in test_dataset])
        train_accuracy = tensorflow.reduce_mean([test_step(x, y) for x, y in train_dataset])
        print(f'Epoch {epoch+1}, loss: {loss:.4f}, test accuracy: {test_accuracy:.4f}, train accuracy: {train_accuracy:.4f}')
