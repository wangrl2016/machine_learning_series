import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras
import numpy
from matplotlib import pyplot
from keras.src.legacy.preprocessing import image

def sample_images(epoch, grid_size=5):
    z = numpy.random.normal(0, 1, (grid_size * grid_size, latent_dim))
    gen_imgs = generator.predict(z)

    # 反归一化到 [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 绘制网格
    fig, axs = pyplot.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    cnt = 0
    for i in range(grid_size):
        for j in range(grid_size):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{output_dir}/{epoch}.png")
    pyplot.close()

def build_generator(latent_dim):
    model = keras.Sequential([
        keras.layers.Dense(8*8*256, input_dim=latent_dim),
        keras.layers.Reshape((8, 8, 256)),
        keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])
    return model

def build_discriminator(img_size):
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(img_size, img_size, 3)),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

if __name__ == '__main__':
    latent_dim = 100
    img_size = 64
    batch_size = 64
    datagen = image.ImageDataGenerator(rescale=1.0/255)
    data = datagen.flow_from_directory(
        '/Users/admin/Downloads/archive/UTKFace/',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=None
    )

    def preprocess_data(images):
        return (images - 0.5) * 2
    dataset = (preprocess_data(batch) for batch in data)

    generator = build_generator(latent_dim)
    optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    generator.compile(optimizer=optimizer, loss='binary_crossentropy')
    generator.summary()
    discriminator = build_discriminator(img_size)
    discriminator.summary()

    discriminator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    # discriminator.trainable = False

    z = keras.layers.Input(shape=(latent_dim,))
    img =generator(z)
    validity = discriminator(img)
    dcgan = keras.Model(z, validity)
    dcgan.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )

    output_dir = 'generated'
    os.makedirs(output_dir, exist_ok=True)

    def train_dcgan(epochs, batch_size, sample_interval):
        real_label = numpy.ones((batch_size, 1))
        fake_label = numpy.zeros((batch_size, 1))

        for epoch in range(epochs):
            for real_imgs in dataset:
                idx = numpy.random.randint(0, real_imgs.shape[0], batch_size)
                real_imgs_batch = real_imgs[idx]
                z = numpy.random.normal(0, 1, (batch_size, latent_dim))
                fake_imgs = generator.predict(z)

                d_loss_real = discriminator.train_on_batch(real_imgs_batch, real_label)
                d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
                d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

                # 训练生成器
                z = numpy.random.normal(0, 1, (batch_size, latent_dim))
                g_loss = dcgan.train_on_batch(z, real_label)
            
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D acc: {100 * d_loss[1]}] [G loss: {g_loss}]")
            if epoch % sample_interval == 0:
                sample_images(epoch)
    
    train_dcgan(epochs=10000, batch_size=64, sample_interval=500)
