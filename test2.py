import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Reshape, Conv2D, Flatten, Dropout,
                                     LeakyReLU, BatchNormalization, Add,
                                     UpSampling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import TensorBoard

# 디렉토리 설정
os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=HeNormal())(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = LeakyReLU(0.2)(x)
    return x

def build_generator(latent_dim):
    input_layer = Input(shape=(latent_dim,))
    x = Dense(128 * 16 * 16, kernel_initializer=HeNormal())(input_layer)
    x = Reshape((16, 16, 128))(x)

    for _ in range(3):
        x = residual_block(x, 128)

    for filters in [128, 64, 32]:
        x = UpSampling2D()(x)
        x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=HeNormal())(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)

    output_layer = Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)
    return Model(input_layer, output_layer, name='generator')

def build_discriminator(image_shape):
    model = Sequential(name='discriminator')
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                     input_shape=image_shape, kernel_initializer=HeNormal()))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    for filters in [128, 256]:
        model.add(Conv2D(filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=HeNormal()))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    model = Model(gan_input, gan_output, name='gan')
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

def load_real_samples(batch_size, dataset_path='dataset'):
    image_files = os.listdir(dataset_path)
    selected_files = random.sample(image_files, min(batch_size, len(image_files)))
    images = []

    for file in selected_files:
        path = os.path.join(dataset_path, file)
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.resize(img, (128, 128))
        img = img / 127.5 - 1
        images.append(img)

    return np.array(images), len(images)

def generate_and_save_images(model, epoch, latent_dim, n_images=25):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    generated = model.predict(noise)
    generated = (generated + 1) / 2.0

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"images/gen_image_epoch_{epoch}.png")
    plt.close()

def train_gan(gan, generator, discriminator, latent_dim, n_epochs=100, n_batch=64):
    half_batch = n_batch // 2
    d_losses, g_losses = [], []

    for epoch in range(n_epochs):
        real_imgs, actual_size = load_real_samples(half_batch)
        real_labels = np.ones((actual_size, 1)) * 0.9  # Label smoothing

        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_imgs = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1)) + np.random.uniform(0, 0.1)

        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (n_batch, latent_dim))
        valid_labels = np.ones((n_batch, 1)) * 0.9
        g_loss = gan.train_on_batch(noise, valid_labels)

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        print(f"Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, latent_dim)
            generator.save(f"saved_models/generator_epoch_{epoch+1}.h5")

    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('images/loss_curve.png')
    plt.close()

# 실행
latent_dim = 100
image_shape = (128, 128, 3)
generator = build_generator(latent_dim)
discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
gan = build_gan(generator, discriminator, latent_dim)

train_gan(gan, generator, discriminator, latent_dim, n_epochs=50, n_batch=64)
