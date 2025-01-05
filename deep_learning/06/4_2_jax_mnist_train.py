# https://github.com/jax-ml/jax/blob/main/examples/mnist_classifier_fromscratch.py

import jax.numpy as jnp
import deep_learning.dataset.mnist as mnist
import numpy
import jax
import time

EPOCHS = 10
BATCH_SIZE = 128
LAYER_SIZES = [784, 512, 256, 10]
STEP = 0.01

if __name__ == '__main__':
    rng = jax.random.key(0)
    (train_images, train_labels), (test_images, test_labels) = mnist.parse()
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)
    train_labels = numpy.eye(10)[train_labels]
    test_labels = numpy.eye(10)[test_labels]
    num_train = train_images.shape[0]
    num_complete_batches, leftover = numpy.divmod(num_train, BATCH_SIZE)
    num_batches = num_complete_batches + bool(leftover)
    # 对数据集进行切片
    def data_stream():
        rng = numpy.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches= data_stream()

    def init_random_params(scale, layer_sizes, rng=numpy.random.RandomState(0)):
        return [(scale * rng.randn(m, n), scale * rng.randn(n))
                for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

    # 前向传播
    def predict(params, inputs):
        activations = inputs
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            # 激活函数
            activations = jnp.tanh(outputs)
        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)

    # 损失函数
    def loss(params, batch):
        inputs, targets = batch
        preds = predict(params, inputs)
        print(preds.shape, targets.shape)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(predict(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    @jax.jit
    def update(params, batch):
        grads = jax.grad(loss)(params, batch)
        return [(w - STEP * dw, b - STEP * db)
                for (w, b), (dw, db) in zip(params, grads)]

    print('Starting training...')
    params = init_random_params(0.1, LAYER_SIZES)
    for epoch in range(EPOCHS):
        start_time = time.time()
        for _ in range(num_batches):
            params = update(params, next(batches))
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f'Epoch {epoch} in {epoch_time:0.2f} sec')
        print(f'Training set accuracy {train_acc}')
        print(f'Test set accuracy {test_acc}')
