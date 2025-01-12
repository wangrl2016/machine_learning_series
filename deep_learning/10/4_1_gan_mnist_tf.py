import time
import tensorflow
import keras
from matplotlib import pyplot
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import mnist

def make_generator_model():
    model = tensorflow.keras.Sequential()
    model.add(keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = tensorflow.keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))

    return model

if __name__ == '__main__':
    print(tensorflow.__version__)

    (train_images, train_labels), (_, _) = mnist.parse()
    print(train_images.shape)
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    # Batch and shuffle the data.
    train_dataset = tensorflow.data.Dataset.from_tensor_slices(
        train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    generator = make_generator_model()
    noise = tensorflow.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    pyplot.imshow(generated_image[0, :, :, 0], cmap='gray')
    pyplot.show()

    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print (decision)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tensorflow.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tensorflow.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tensorflow.ones_like(fake_output), fake_output)
    
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)

    checkpoint_dir = './temp/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tensorflow.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tensorflow.random.normal([num_examples_to_generate, noise_dim])

    # Notice the use of `tensorflow.function`
    # This annotation causes the function to be compiled.
    @tensorflow.function
    def train_step(images):
        noise = tensorflow.random.normal([BATCH_SIZE, noise_dim])

        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return gen_loss, disc_loss
    
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)
        pyplot.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            pyplot.subplot(4, 4, i+1)
            pyplot.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            pyplot.axis('off')
        
        pyplot.savefig('temp/image_at_epoch_{:04d}.png'.format(epoch))
        # pyplot.show()

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for batch_num, image_batch in enumerate(dataset):
                generator_loss, discriminator_loss = train_step(image_batch)

                epoch_elapsed_time = int(time.time() - start)
                print(f"\rEpoch {epoch+1}/{epochs}:  "\
                        f"Elapsed time: {epoch_elapsed_time}s  "\
                        f"Batch {batch_num}/{BUFFER_SIZE // BATCH_SIZE }:  "\
                        f"generator_loss: {generator_loss:<6.6f} "\
                        f"discriminator_loss: {discriminator_loss/2:<6.6f}", end="")
            print()

            generate_and_save_images(generator,
                                     epoch+1,
                                     seed)
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        generate_and_save_images(generator, epochs, seed)

    train(train_dataset, EPOCHS)
