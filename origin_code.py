import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Reshape, Conv2D, Flatten, Dropout,
                                     LeakyReLU, BatchNormalization, Add,
                                     UpSampling2D)
from tensorflow.keras.optimizers import Adam

# 이미지 저장 폴더 생성
os.makedirs('images', exist_ok=True)

def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = LeakyReLU(0.2)(x)
    return x

def build_generator(latent_dim):
    model_input = Input(shape=(latent_dim,))
    x = Dense(128 * 16 * 16)(model_input)
    x = Reshape((16, 16, 128))(x)

    x = residual_block(x, 128)

    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    out_image = Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)
    return Model(model_input, out_image, name='generator')

def build_discriminator(image_shape):
    from tensorflow.keras.models import Sequential
    model = Sequential(name='discriminator')
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output, name='gan')
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

def load_real_samples(batch_size, dataset_path='dataset'):
    image_files = os.listdir(dataset_path)
    n_images = len(image_files)
    batch_size = min(batch_size, n_images)
    selected_files = random.sample(image_files, batch_size)
    images = []
    for file in selected_files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = img * 2 - 1
        images.append(img)
    images = np.array(images)
    return images, images.shape[0]

def train_gan(gan, generator, discriminator, latent_dim, n_epochs=100, n_batch=128):

    def safe_loss(loss):
        try:
            return loss[0]
        except (TypeError, IndexError):
            return loss

    half_batch = n_batch // 2
    d_losses, g_losses = [], []

    for epoch in range(n_epochs):
        real_images, actual_size = load_real_samples(half_batch)
        real_labels = np.ones((actual_size, 1))
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        d_loss = 0.5 * (safe_loss(d_loss_real) + safe_loss(d_loss_fake))

        noise = np.random.normal(0, 1, (n_batch, latent_dim))
        gan_labels = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(noise, gan_labels)

        print(f"Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss:.4f}, G Loss: {safe_loss(g_loss):.4f}")
        d_losses.append(d_loss)
        g_losses.append(safe_loss(g_loss))

        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, latent_dim)

    # 학습곡선 시각화
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.savefig('images/loss_curve.png')
    plt.close()

def generate_and_save_images(model, epoch, latent_dim, n_images=25):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    generated_images = model.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"images/generated_image_epoch_{epoch}.png")
    plt.close()

# 모델 구성 및 학습 시작
latent_dim = 100
image_shape = (128, 128, 3)
generator = build_generator(latent_dim)
discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
gan = build_gan(generator, discriminator, latent_dim)
train_gan(gan, generator, discriminator, latent_dim, n_epochs=100, n_batch=128)
