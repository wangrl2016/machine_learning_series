# Make sure to set the environment variable before import keras
import os
os.environ["KERAS_BACKEND"] = "jax"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import mnist
from matplotlib import pyplot

import keras
from keras import layers, ops, random
import jax
import jax.numpy as jnp
import time
import numpy

def batch_and_shuffle(data, batch_size, shuffle=True):
    # 随机打乱数据索引
    if shuffle:
        numpy.random.shuffle(data)
    
    # 按批次切分数据
    num_batches = len(data) // batch_size
    batches = []
    
    for i in range(num_batches):
        batch_data = data[i * batch_size : (i + 1) * batch_size]
        batches.append(batch_data)

    return numpy.array(batches)

def get_generator(image_shape: tuple = (28, 28, 1),
                    latent_dim: int = 100):
  inputs = layers.Input(shape=(latent_dim,), name="generator_input")
  x = layers.Dense(256)(inputs)
  x = layers.LeakyReLU(negative_slope=0.2)(x)
  x = layers.BatchNormalization(momentum=0.5)(x)
  x = layers.Dense(512)(x)
  x = layers.LeakyReLU(negative_slope=0.2)(x)
  x = layers.BatchNormalization(momentum=0.5)(x)
  x = layers.Dense(1024)(x)
  x = layers.LeakyReLU(negative_slope=0.2)(x)
  x = layers.BatchNormalization(momentum=0.5)(x)
  x = layers.Dense(ops.prod(jnp.array(image_shape)).item(), activation="tanh")(x)
  outputs = layers.Reshape(image_shape, name="generator_output")(x)

  return keras.Model(inputs, outputs, name="generator")

def get_discriminator(image_shape: tuple = (28, 28, 1)):
  inputs = layers.Input(shape=image_shape, name="discriminator_input")
  x = layers.Flatten()(inputs)
  x = layers.Dense(512)(x)
  x = layers.LeakyReLU(negative_slope=0.2)(x)
  x = layers.Dense(256)(x)
  x = layers.LeakyReLU(negative_slope=0.2)(x)
  x = layers.Dense(128)(x)
  x = layers.LeakyReLU(negative_slope=0.2)(x)
  outputs = layers.Dense(1, activation="sigmoid",
                         name="discriminator_output")(x)

  return keras.Model(inputs, outputs, name="discriminator")

if __name__ == '__main__':
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
    LATENT_DIM = 100
    (mnist_training_data, _), (_, _) = mnist.parse()

    # scale the data to [-1, 1] range
    # Since the original values are in range [0, 255],
    # so to scale the values to [-1, 1]
    # we first subtract 255/2=127.5 and then divide by it as well
    mnist_training_data = (mnist_training_data - 127.5) / 127.5

    NUM_SAMPLES = len(mnist_training_data)

    mnist_dataset = batch_and_shuffle(mnist_training_data, BATCH_SIZE, shuffle=True)
    print(mnist_dataset.shape)

    generator = get_generator(latent_dim=LATENT_DIM)
    discriminator = get_discriminator()
    generator.summary()
    discriminator.summary()

    loss_fn = keras.losses.BinaryCrossentropy()

    generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002,
                                                beta_1=0.5)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002,
                                                beta_1=0.5)
    

    # Make sure to build optimizer variables
    generator_optimizer.build(generator.trainable_variables)
    discriminator_optimizer.build(discriminator.trainable_variables)

    def generator_compute_loss_and_updates(gen_train_var,
                                           gen_non_train_var,
                                           dis_train_var,
                                           dis_non_train_var,
                                           noise):
        fake_images, gen_non_train_var = generator.stateless_call(
            gen_train_var,
            gen_non_train_var,
            noise,
            training=True)
        y_pred, dis_non_train_var = discriminator.stateless_call(
            dis_train_var,
            dis_non_train_var,
            fake_images,
            training=True)
        generator_loss = loss_fn(y_true=ops.ones(ops.shape(y_pred)),
                                 y_pred=y_pred)
        return generator_loss, gen_non_train_var


    generator_grad_fn = jax.value_and_grad(generator_compute_loss_and_updates, has_aux=True)

    @jax.jit
    def generator_train_step(generator_state: tuple,
                            discriminator_state: tuple,
                            noise):
        (
            generator_trainable_variables,
            generator_non_trainable_variables,
            generator_optimizer_variables
        ) = generator_state

        (
            discriminator_trainable_variables,
            discriminator_non_trainable_variables,
            _
        ) = discriminator_state

        (loss, generator_non_trainable_variables), grads = generator_grad_fn(
                                        generator_trainable_variables,
                                        generator_non_trainable_variables,
                                        discriminator_trainable_variables,
                                        discriminator_non_trainable_variables,
                                        noise)
        # Optimize
        (
            generator_trainable_variables,
            generator_optimizer_variables
        ) = generator_optimizer.stateless_apply(generator_optimizer_variables,
                                                grads,
                                                generator_trainable_variables)

        # Since generator_train_step function does not modify the discriminator state
        # and only updates the generator state so we return the updated state
        generator_state = (generator_trainable_variables,
                            generator_non_trainable_variables,
                            generator_optimizer_variables)
        return loss, generator_state
    

    def discriminator_compute_loss_and_updates(discriminator_trainable_variables,
                                            discriminator_non_trainable_variables,
                                            real_images,
                                            fake_images,
                                            noise,
                                            ):
        # train on fake images
        y_pred_fake, discriminator_non_trainable_variables = discriminator.stateless_call(
                                                discriminator_trainable_variables,
                                                discriminator_non_trainable_variables,
                                                fake_images,
                                                training=True)
        discriminator_loss_fake = loss_fn(y_true=ops.zeros(ops.shape(y_pred_fake)),
                                            y_pred=y_pred_fake)

        # train on real images
        y_pred_real, discriminator_non_trainable_variables = discriminator.stateless_call(
                                                discriminator_trainable_variables,
                                                discriminator_non_trainable_variables,
                                                real_images,
                                                training=True)
        discriminator_loss_real = loss_fn(y_true=ops.ones(ops.shape(y_pred_real)),
                                            y_pred=y_pred_real)

        discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        return discriminator_loss, discriminator_non_trainable_variables


    discriminator_grad_fn = jax.value_and_grad(
                                        discriminator_compute_loss_and_updates,
                                        has_aux=True)
    
    @jax.jit
    def discriminator_train_step(discriminator_state: tuple,
                                generator_state: tuple,
                                real_images,
                                noise):
        (
            discriminator_trainable_variables,
            discriminator_non_trainable_variables,
            discriminator_optimizer_variables
            ) = discriminator_state

        (
            generator_trainable_variables,
            generator_non_trainable_variables,
            _
            ) = generator_state

        fake_images, _ = generator.stateless_call(
                                                generator_trainable_variables,
                                                generator_non_trainable_variables,
                                                noise,
                                                training=False)
        (loss, discriminator_non_trainable_variables), grads = discriminator_grad_fn(
                                  discriminator_trainable_variables,
                                  discriminator_non_trainable_variables,
                                  real_images,
                                  fake_images,
                                  noise
                                  )
        # Optimize
        (discriminator_trainable_variables,
         discriminator_optimizer_variables
         ) = discriminator_optimizer.stateless_apply(
                                                    discriminator_optimizer_variables,
                                                    grads,
                                                    discriminator_trainable_variables)
        # Since discriminator_train_step function does not modify the
        # generator state rather only updates the discriminator state so
        # we return the updated discriminator state
        discriminator_state = (discriminator_trainable_variables,
                                discriminator_non_trainable_variables,
                                discriminator_optimizer_variables)
        return loss, discriminator_state
    

    # get the initial state
    generator_state = (generator.trainable_variables,
                       generator.non_trainable_variables,
                       generator_optimizer.variables)

    discriminator_state = (discriminator.trainable_variables,
                        discriminator.non_trainable_variables,
                        discriminator_optimizer.variables)
    
    temp = NUM_SAMPLES // BATCH_SIZE
    TOTAL_BATCHES = (temp + 0) if NUM_SAMPLES % BATCH_SIZE == 0 else (temp + 1)

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        for batch_num, real_images in enumerate(mnist_dataset):
            noise = random.normal((BATCH_SIZE, LATENT_DIM))
            generator_loss, generator_state = generator_train_step(
                                                                generator_state,
                                                                discriminator_state,
                                                                noise)
            discriminator_loss, discriminator_state = discriminator_train_step(
                                                                discriminator_state,
                                                                generator_state,
                                                                real_images,
                                                                noise)
            # Divide discriminator loss by 2 to get the average loss since we sum up
            # the discrimintor loss of real and fake samples
            epoch_elapsed_time = int(time.time() - epoch_start_time)
            print(f"\rEpoch {epoch+1}/{NUM_EPOCHS}:  "\
                f"Elapsed time: {epoch_elapsed_time}s  "\
                f"Batch {batch_num+1}/{TOTAL_BATCHES}:  "\
                f"generator_loss: {generator_loss:<6.6f} "\
                f"discriminator_loss: {discriminator_loss/2:<6.6f}", end="")
        print()
    
    def generate_images(generator, generator_state, noise):
        trainable_vars, non_trainable_vars, _ = generator_state
        predictions, _ = generator.stateless_call(trainable_vars,
                                                    non_trainable_vars,
                                                    noise,
                                                    training=False)

        pyplot.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            pyplot.subplot(4, 4, i+1)
            pyplot.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            pyplot.axis('off')
        pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
        pyplot.show()

    for _ in range(5):
        noise = random.normal((16, LATENT_DIM))
        generate_images(generator, generator_state, noise)
