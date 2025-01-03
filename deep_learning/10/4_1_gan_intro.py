import keras
import tensorflow
import time
from matplotlib import pyplot


BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

def make_generator_model():
    model = keras.Sequential()
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
    model = keras.Sequential()
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

def generator_loss(loss, fake_output):
    return loss(tensorflow.ones_like(fake_output), fake_output)

def discriminator_loss(loss, real_output, fake_output):
    real_loss = loss(tensorflow.ones_like(real_output), real_output)
    fake_loss = loss(tensorflow.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

@tensorflow.function
def train_step(images):
    noise = tensorflow.random.normal([BATCH_SIZE, noise_dim])
    
    with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(cross_entropy, fake_output)
        disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = pyplot.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        pyplot.subplot(4, 4, i+1)
        pyplot.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        pyplot.axis('off')

    pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    pyplot.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        generate_and_save_images(generator, epoch + 1, seed)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)

if __name__ == '__main__':
    print(tensorflow.__version__)
    (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    # Batch and shuffle the data
    train_dataset = tensorflow.data.Dataset.from_tensor_slices(
        train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    generator = make_generator_model()
    noise = tensorflow.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    pyplot.imshow(generated_image[0, :, :, 0], cmap='gray')
    pyplot.show()
    
    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print(decision)
    
    # This method returns a helper function to compute corss entropy loss.
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)
    
    seed = tensorflow.random.normal([num_examples_to_generate, noise_dim])
    
    train(train_dataset, EPOCHS)

    
    
    