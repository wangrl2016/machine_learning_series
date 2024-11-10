# https://github.com/jax-ml/jax/blob/main/examples/mnist_classifier_fromscratch.py

import jax.numpy as jnp
import numpy
import jax
import time
import mnist_parse

EPOCHS = 10
BATCH_SIZE = 128

if __name__ == '__main__':
    rng = jax.random.key(0)

    (train_images, train_labels), (test_images, test_labels) = mnist_parse.dataset()
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

    print('Starting training...')
    for epoch in range(EPOCHS):
        start_time = time.time()
        for _ in range(num_batches):
            pass
