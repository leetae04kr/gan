import cv2
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import random

# 이미지 저장 폴더가 없으면 생성
os.makedirs('images', exist_ok=True)
# 1. 이미지 전처리 함수
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # 이미지 정규화 (0 ~ 1)
    return image

# 생성자 모델 구축 함수
def build_generator(latent_dim):
    model = Sequential(name='generator')
    model.add(Dense(128 * 16 * 16, input_dim=latent_dim))
    model.add(Reshape((16, 16, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'))
    return model

# 생성자 모델 초기화
latent_dim = 100
generator = build_generator(latent_dim)

# 판별자(Discriminator) 모델 구축 함수
def build_discriminator(image_shape):
    model = Sequential(name='discriminator')
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))  # feature map 줄이기
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))  # 추가 (128x128은 깊은 feature 필요)
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


# 판별자 모델 초기화
image_shape = (128, 128, 3)  # 변경
discriminator = build_discriminator(image_shape)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])


# GAN 모델 구축 함수
def build_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False  # 판별자는 GAN 학습 시에는 학습하지 않음
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output, name='gan')
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

# GAN 모델 초기화
gan = build_gan(generator, discriminator, latent_dim)


def load_real_samples(batch_size, dataset_path='dataset'):# <--여기 
    # dataset 폴더에 있는 이미지 파일 리스트
    image_files = os.listdir(dataset_path)

    # 이미지 개수 확인
    n_images = len(image_files)

    # 데이터 부족 대비
    batch_size = min(batch_size, n_images)

    # 랜덤하게 batch_size만큼 이미지 선택
    selected_files = random.sample(image_files, batch_size)

    images = []
    for file in selected_files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue  # 파일 읽기 실패하면 무시
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 읽기 때문에 RGB 변환
        img = cv2.resize(img, (128, 128))  # 반드시 128x128
        img = img / 255.0  # 0~1 정규화
        img = img * 2 - 1  # [-1, 1] 범위로 변환
        images.append(img)

    images = np.array(images)

    return images


# 모델 학습
def train_gan(gan, generator, discriminator, latent_dim, n_epochs=1000, n_batch=128):
    half_batch = int(n_batch / 2)
    for epoch in range(n_epochs):
        # 판별자 학습
        real_images = load_real_samples(half_batch)
        real_labels = np.ones((half_batch, 1))
        discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        # 손실 값만 추출
        d_loss_real_value = discriminator_loss_real[0] if isinstance(discriminator_loss_real, (list, np.ndarray)) else discriminator_loss_real
        d_loss_fake_value = discriminator_loss_fake[0] if isinstance(discriminator_loss_fake, (list, np.ndarray)) else discriminator_loss_fake
        d_loss = 0.5 * (d_loss_real_value + d_loss_fake_value)

        # 생성자 학습 (GAN 모델을 통해?)
        noise = np.random.normal(0, 1, (n_batch, latent_dim))
        gan_labels = np.ones((n_batch, 1))

        gan_loss = gan.train_on_batch(noise, gan_labels)
        gan_loss_value = gan_loss[0] if isinstance(gan_loss, (list, np.ndarray)) else gan_loss

        # 출력
        print(f"Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss:.4f}, G Loss: {gan_loss_value:.4f}")

        # 일정 간격마다 이미지 저장
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, latent_dim)


def generate_fake_images(generator, latent_dim, n_images):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    fake_images = generator.predict(noise)
    return fake_images

def display_images(images, rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i, img in enumerate(images):
        img = (img + 1) / 2.0  # tanh 활성화 함수로 인해 -1 ~ 1 사이의 값을 0 ~ 1로 조정
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def generate_and_save_images(model, epoch, latent_dim, n_images=25):

    noise = np.random.normal(0, 1, (n_images, latent_dim))
    generated_images = model.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # tanh 활성화 함수로 인해 -1 ~ 1 사이의 값을 0 ~ 1로 조정

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"images/generated_image_epoch_{epoch}.png")
    plt.close()
train_gan(gan, generator, discriminator, latent_dim, n_epochs=1000, n_batch=128)
