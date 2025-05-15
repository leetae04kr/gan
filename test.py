# 필수 라이브러리 불러오기
import cv2  # 이미지 읽기 및 전처리용 OpenCV
import numpy as np  # 수치 계산
from tensorflow.keras.models import Sequential, Model  # 모델 구조 정의용
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Input  # 층들
from tensorflow.keras.optimizers import Adam  # 최적화 알고리즘
import matplotlib.pyplot as plt  # 시각화용
import os  # 파일/폴더 처리
import random  # 데이터 무작위 추출

# 생성된 이미지를 저장할 폴더가 없다면 자동 생성
os.makedirs('images', exist_ok=True)

def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)  # 이미지 로드 (BGR)
    image = cv2.resize(image, target_size)  # 크기 조정
    image = image / 255.0  # [0, 255] → [0, 1] 정규화
    return image

def build_generator(latent_dim):
    model = Sequential(name='generator')

    # 입력 벡터 (latent vector)를 128채널, 16x16 텐서로 변형
    model.add(Dense(128 * 16 * 16, input_dim=latent_dim))
    model.add(Reshape((16, 16, 128)))  # (None, 32768) → (None, 16, 16, 128)

    # 업샘플링: 16x16 → 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))  # 음수 값도 일부 통과시켜 학습 안정화
    model.add(BatchNormalization(momentum=0.8))  # 내부 공변산성 정규화

    # 업샘플링: 32x32 → 64x64
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # 업샘플링: 64x64 → 128x128, 출력 채널 수는 3 (RGB)
    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'))

    return model

latent_dim = 100  # 잠재 벡터 z의 차원 (노이즈 벡터)
generator = build_generator(latent_dim)  # 생성자 모델 생성

def build_discriminator(image_shape):
    model = Sequential(name='discriminator')

    # Conv Block 1: 128x128 → 64x64
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))  # 과적합 방지용 드롭아웃

    # Conv Block 2: 64x64 → 32x32
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    # Conv Block 3: 32x32 → 16x16
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())  # 벡터 형태로 변형
    model.add(Dense(1, activation='sigmoid'))  # 0(가짜) 또는 1(진짜) 확률 출력

    return model

image_shape = (128, 128, 3)  # 입력 이미지의 크기
discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])  # 컴파일

def build_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False  # 생성자 학습 시 판별자는 고정

    gan_input = Input(shape=(latent_dim,))  # 노이즈 입력
    gan_output = discriminator(generator(gan_input))  # 생성된 이미지를 판별기에 통과

    gan = Model(gan_input, gan_output, name='gan')  # 생성자+판별자 연결 모델
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))  # GAN 모델은 오직 생성자만 학습

    return gan

gan = build_gan(generator, discriminator, latent_dim)


def load_real_samples(batch_size, dataset_path='gan/dataset'):
    image_files = os.listdir(dataset_path)
    n_images = len(image_files)
    batch_size = min(batch_size, n_images)  # 실제 이미지 수보다 큰 배치 방지

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
        img = img * 2 - 1  # [0,1] → [-1,1]로 정규화 (tanh 출력과 맞춤)
        images.append(img)

    images = np.array(images)
    actual_size = images.shape[0]  # 불러온 이미지 수
    return images, actual_size

# def train_gan(gan, generator, discriminator, latent_dim, n_epochs=1000, n_batch=128):
#     def safe_extract_loss(loss):
#         # train_on_batch 결과가 리스트일 수 있으므로 안전하게 처리
#         if isinstance(loss, (list, np.ndarray)) and np.ndim(loss) > 0:
#             return loss[0]
#         return loss

#     half_batch = int(n_batch / 2)  # 판별자는 절반만 사용 (진짜/가짜 각각)

#     for epoch in range(n_epochs):
#         # ----- 진짜 이미지 -----
#         real_images, actual_size = load_real_samples(half_batch)
#         real_labels = np.ones((actual_size, 1))  # 진짜 라벨은 1
#         discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)

#         # ----- 가짜 이미지 -----
#         noise = np.random.normal(0, 1, (half_batch, latent_dim))  # 노이즈 생성
#         fake_images = generator.predict(noise)
#         fake_labels = np.zeros((half_batch, 1))  # 가짜 라벨은 0
#         discriminator_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

#         # 판별자 손실 평균 계산
#         d_loss_real_value = safe_extract_loss(discriminator_loss_real)
#         d_loss_fake_value = safe_extract_loss(discriminator_loss_fake)
#         d_loss = 0.5 * (d_loss_real_value + d_loss_fake_value)

#         # ----- 생성자 학습 (GAN) -----
#         noise = np.random.normal(0, 1, (n_batch, latent_dim))  # 새 노이즈
#         gan_labels = np.ones((n_batch, 1))  # 생성자 학습용 라벨은 "진짜(1)"
#         gan_loss = gan.train_on_batch(noise, gan_labels)
#         gan_loss_value = safe_extract_loss(gan_loss)

#         # 훈련 결과 출력
#         print(f"Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss:.4f}, G Loss: {gan_loss_value:.4f}")

#         # 일정 epoch마다 이미지 저장
#         if (epoch + 1) % 10 == 0:
#             generate_and_save_images(generator, epoch + 1, latent_dim)
def train_gan(gan, generator, discriminator, latent_dim, n_epochs=1000, n_batch=128):
    def safe_extract_loss(loss):
        if isinstance(loss, (list, np.ndarray)) and np.ndim(loss) > 0:
            return loss[0]
        return loss

    half_batch = int(n_batch / 2)

    # 손실 기록용 리스트
    d_losses = []
    g_losses = []

    for epoch in range(n_epochs):
        real_images, actual_size = load_real_samples(half_batch)
        real_labels = np.ones((actual_size, 1))
        discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        d_loss_real_value = safe_extract_loss(discriminator_loss_real)
        d_loss_fake_value = safe_extract_loss(discriminator_loss_fake)
        d_loss = 0.5 * (d_loss_real_value + d_loss_fake_value)

        noise = np.random.normal(0, 1, (n_batch, latent_dim))
        gan_labels = np.ones((n_batch, 1))
        gan_loss = gan.train_on_batch(noise, gan_labels)
        gan_loss_value = safe_extract_loss(gan_loss)

        # 손실 기록
        d_losses.append(d_loss)
        g_losses.append(gan_loss_value)

        print(f"Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss:.4f}, G Loss: {gan_loss_value:.4f}")

        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, latent_dim)

    # 학습 완료 후 손실 시각화
    plot_loss(d_losses, g_losses)


def plot_loss(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/loss_curve.png')  # 저장
    plt.show()


# 가짜 이미지 생성
def generate_fake_images(generator, latent_dim, n_images):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    fake_images = generator.predict(noise)
    return fake_images

# 생성된 이미지를 화면에 표시
def display_images(images, rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i, img in enumerate(images):
        img = (img + 1) / 2.0  # [-1,1] → [0,1]
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# 이미지 저장 (파일로)
def generate_and_save_images(model, epoch, latent_dim, n_images=25):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    generated_images = model.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # [-1,1] → [0,1]

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"gan/images/generated_image_epoch_{epoch}.png")
    plt.close()


train_gan(gan, generator, discriminator, latent_dim, n_epochs=50, n_batch=128)