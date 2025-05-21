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

# 이미지 저장 폴더 생성 (존재하면 무시)
os.makedirs('images', exist_ok=True)

# 이미지 전처리 함수: 읽고, 크기 조정, 정규화
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)  # OpenCV를 사용하여 이미지 읽기
    image = cv2.resize(image, target_size)  # 이미지 크기를 target_size로 조정
    image = image / 255.0  # 픽셀 값을 0과 1 사이로 정규화
    return image

# ResNet 스타일의 잔차 블록 정의
def residual_block(x, filters):
    shortcut = x  # 입력값을 shortcut으로 저장
    x = Conv2D(filters, kernel_size=3, padding='same')(x)  # 3x3 컨볼루션 레이어
    x = LeakyReLU(0.2)(x)  # LeakyReLU 활성화 함수
    x = BatchNormalization()(x)  # 배치 정규화 레이어
    x = Conv2D(filters, kernel_size=3, padding='same')(x)  # 또 다른 3x3 컨볼루션 레이어
    x = BatchNormalization()(x)  # 배치 정규화 레이어
    x = Add()([x, shortcut])  # shortcut과 컨볼루션 결과를 더함 (잔차 연결)
    x = LeakyReLU(0.2)(x)  # LeakyReLU 활성화 함수
    return x

# 생성자 모델 구축
def build_generator(latent_dim):
    model_input = Input(shape=(latent_dim,))  # 잠재 공간 벡터 입력
    x = Dense(128 * 16 * 16)(model_input)  # 완전 연결 레이어, 초기 특징 맵 생성
    x = Reshape((16, 16, 128))(x)  # 특징 맵을 16x16x128 형태로 재구성

    x = residual_block(x, 128)  # 잔차 블록 추가

    x = UpSampling2D()(x)  # 2배 업샘플링
    x = Conv2D(128, kernel_size=3, padding='same')(x)  # 컨볼루션 레이어
    x = LeakyReLU(0.2)(x)  # LeakyReLU 활성화 함수
    x = BatchNormalization()(x)  # 배치 정규화 레이어

    x = UpSampling2D()(x)  # 2배 업샘플링
    x = Conv2D(64, kernel_size=3, padding='same')(x)  # 컨볼루션 레이어
    x = LeakyReLU(0.2)(x)  # LeakyReLU 활성화 함수
    x = BatchNormalization()(x)  # 배치 정규화 레이어

    x = UpSampling2D()(x)  # 2배 업샘플링
    x = Conv2D(32, kernel_size=3, padding='same')(x)  # 컨볼루션 레이어
    x = LeakyReLU(0.2)(x)  # LeakyReLU 활성화 함수
    x = BatchNormalization()(x)  # 배치 정규화 레이어

    out_image = Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)  # 최종 컨볼루션 레이어, tanh 활성화 함수로 픽셀 값을 -1과 1 사이로 조정
    return Model(model_input, out_image, name='generator')  # 입력과 출력을 연결하여 모델 생성

# 판별자 모델 구축
def build_discriminator(image_shape):
    from tensorflow.keras.models import Sequential  # Sequential 모델 사용
    model = Sequential(name='discriminator')
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))  # 컨볼루션 레이어, 스트라이드 2로 특징 맵 크기 축소
    model.add(LeakyReLU(0.2))  # LeakyReLU 활성화 함수
    model.add(Dropout(0.4))  # 드롭아웃 레이어 (과적합 방지)

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))  # 컨볼루션 레이어
    model.add(LeakyReLU(0.2))  # LeakyReLU 활성화 함수
    model.add(Dropout(0.4))  # 드롭아웃 레이어

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))  # 컨볼루션 레이어
    model.add(LeakyReLU(0.2))  # LeakyReLU 활성화 함수
    model.add(Dropout(0.4))  # 드롭아웃 레이어

    model.add(Flatten())  # 특징 맵을 1차원 벡터로 펼침
    model.add(Dense(1, activation='sigmoid'))  # 완전 연결 레이어, sigmoid 활성화 함수로 0 또는 1의 확률 출력 (진짜/가짜)
    return model

# GAN 모델 구축 (생성자와 판별자 연결)
def build_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False  # 판별자의 학습을 비활성화 (GAN 모델 학습 시 생성자만 학습)
    gan_input = Input(shape=(latent_dim,))  # GAN의 입력은 잠재 공간 벡터
    gan_output = discriminator(generator(gan_input))  # 생성자가 생성한 이미지를 판별자에 입력
    gan = Model(gan_input, gan_output, name='gan')  # 입력과 출력을 연결하여 GAN 모델 생성
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))  # 손실 함수와 최적화 알고리즘 설정
    return gan

# 실제 이미지 데이터 로드 및 전처리
def load_real_samples(batch_size, dataset_path='/Users/leetae04kr/folder/testfolder/dataset'):
    image_files = os.listdir(dataset_path)  # 데이터셋 폴더의 모든 파일 목록 가져오기
    n_images = len(image_files)  # 총 이미지 수
    batch_size = min(batch_size, n_images)  # 배치 크기가 이미지 수보다 크지 않도록 조정
    selected_files = random.sample(image_files, batch_size)  # 무작위로 배치 크기만큼 파일 선택
    images = []
    for file in selected_files:
        img_path = os.path.join(dataset_path, file)  # 이미지 파일의 전체 경로 생성
        img = cv2.imread(img_path)  # 이미지 읽기
        if img is None:  # 이미지 로드 실패 시 건너뛰기
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 색상 공간 변환
        img = cv2.resize(img, (128, 128))  # 이미지 크기 조정
        img = img / 255.0  # 픽셀 값 정규화 (0 ~ 1)
        img = img * 2 - 1  # 픽셀 값 범위 조정 (-1 ~ 1, 생성자 출력과 맞춤)
        images.append(img)
    images = np.array(images)  # 리스트를 numpy 배열로 변환
    return images, images.shape[0]  # 이미지 배열과 실제 배치 크기 반환

# GAN 모델 학습
def train_gan(gan, generator, discriminator, latent_dim, n_epochs=100, n_batch=128):

    # 손실 값 추출 시 오류 방지 함수
    def safe_loss(loss):
        try:
            return loss[0]  # Keras의 train_on_batch는 손실과 메트릭을 함께 반환할 수 있음
        except (TypeError, IndexError):
            return loss  # 손실만 반환된 경우 그대로 반환

    half_batch = n_batch // 2  # 각 학습 단계에서 사용할 배치 절반 크기
    d_losses, g_losses = [], []  # 판별자와 생성자의 손실 기록 리스트

    for epoch in range(n_epochs):  # 에포크 수만큼 반복
        # 실제 이미지 배치 학습
        real_images, actual_size = load_real_samples(half_batch)  # 실제 이미지 로드
        real_labels = np.ones((actual_size, 1))  # 실제 이미지 레이블 (1)
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)  # 판별자 학습

        # 가짜 이미지 배치 학습
        noise = np.random.normal(0, 1, (half_batch, latent_dim))  # 무작위 노이즈 생성
        fake_images = generator.predict(noise)  # 생성자로 가짜 이미지 생성
        fake_labels = np.zeros((half_batch, 1))  # 가짜 이미지 레이블 (0)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)  # 판별자 학습

        # 판별자 전체 손실 계산
        d_loss = 0.5 * (safe_loss(d_loss_real) + safe_loss(d_loss_fake))

        # 생성자 학습
        noise = np.random.normal(0, 1, (n_batch, latent_dim))  # 무작위 노이즈 생성
        gan_labels = np.ones((n_batch, 1))  # 생성자가 생성한 이미지를 실제 이미지로 속이도록 학습 (레이블 1)
        g_loss = gan.train_on_batch(noise, gan_labels)  # GAN 모델 학습 (생성자 업데이트)

        # 현재 에포크의 손실 값 출력
        print(f"Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss:.4f}, G Loss: {safe_loss(g_loss):.4f}")
        d_losses.append(d_loss)  # 판별자 손실 기록
        g_losses.append(safe_loss(g_loss))  # 생성자 손실 기록

        # 10 에포크마다 생성된 이미지 저장
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, latent_dim)

    # 학습 곡선 시각화
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.savefig('images/loss_curve.png')  # 손실 곡선 이미지로 저장
    plt.close()

# 생성된 이미지 저장 함수
def generate_and_save_images(model, epoch, latent_dim, n_images=25):
    noise = np.random.normal(0, 1, (n_images, latent_dim))  # 무작위 노이즈 생성
    generated_images = model.predict(noise)  # 생성자로 이미지 생성
    generated_images = 0.5 * generated_images + 0.5  # 픽셀 값을 0과 1 사이로 스케일링

    # 생성된 이미지 시각화 및 저장
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated_images[cnt])  # 이미지 표시
            axs[i, j].axis('off')  # 축 숨기기
            cnt += 1
    fig.savefig(f"images/generated_image_epoch_{epoch}.png")  # 생성된 이미지 파일로 저장
    plt.close()

# 모델 구성 및 학습 시작
latent_dim = 100  # 잠재 공간의 차원
image_shape = (128, 128, 3)  # 입력 이미지 형태 (높이, 너비, 채널)
generator = build_generator(latent_dim)  # 생성자 모델 생성
discriminator = build_discriminator(image_shape)  # 판별자 모델 생성
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])  # 판별자 컴파일
gan = build_gan(generator, discriminator, latent_dim)  # GAN 모델 생성
train_gan(gan, generator, discriminator, latent_dim, n_epochs=50, n_batch=128)  # GAN 학습 시작