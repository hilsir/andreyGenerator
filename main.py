import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.noise_size = 100  # Размерность шума
        self.img_channels = 3   # 3 канала для RGB
        self.img_height = 600   # Фиксированная высота
        self.img_width = 400    # Фиксированная ширина

        # Используем 4 увеличения, поэтому делим на 16 (2^4)
        self.init_height = self.img_height // 16  # 600 / 16 = 37.5 -> округляем до 37
        self.init_width = self.img_width // 16    # 400 / 16 = 25

        # fc - fully connected (полносвязный слой)
        self.fc_noise = nn.Sequential(
            nn.Linear(self.noise_size, 128 * self.init_height * self.init_width),  # Преобразует шум в тензор
            nn.LayerNorm(128 * self.init_height * self.init_width),              # Нормализация
            nn.LeakyReLU(0.2, inplace=True)                                      # Активация
        )

        # Сверточные блоки
        self.conv_blocks = nn.Sequential(
            # Блок 1: Увеличение с 37x25 до 74x50
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 2: Увеличение с 74x50 до 148x100
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 3: Увеличение с 148x100 до 296x200
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 4: Увеличение с 296x200 до 592x400
            nn.ConvTranspose2d(64, self.img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Финальная активация
        )

        # Интерполяция до 600x400 (назовём это магией для округления)
        self.upsample = nn.Upsample(size=(600, 400), mode='bilinear', align_corners=False)

    def forward(self, z):
        # Преобразуем шум в одномерный тензор
        noise_vector = self.fc_noise(z)
        # Изменяем форму тензора для сверточных операций на 128 потоков
        noise = noise_vector.view(noise_vector.shape[0], 128, self.init_height, self.init_width)
        # Проходим через сверточные блоки
        img = self.conv_blocks(noise)
        # Интерполяция до 600x400
        img = self.upsample(img)
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Размеры изображения
        self.img_channels = 3
        self.img_height = 600
        self.img_width = 400

        # Сверточные блоки
        self.conv_blocks = nn.Sequential(
            # Блок 1: Уменьшение с 600x400 до 300x200
            nn.Conv2d(self.img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 2: Уменьшение с 300x200 до 150x100
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 3: Уменьшение с 150x100 до 75x50
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 4: Уменьшение с 75x50 до 38x25
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 5: Уменьшение с 38x25 до 19x13
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
        )

        # Полносвязный слой для получения одного числа
        self.fc = nn.Linear(18 * 12, 1)

    def forward(self, x):
        # Проходим через сверточные блоки
        x = self.conv_blocks(x)
        print(f"Размер после сверточных слоев: {x.shape}")  # Отладочный вывод
        # Изменяем форму тензора для полносвязного слоя
        x = x.view(x.size(0), -1)
        print(f"Размер перед полносвязным слоем: {x.shape}")  # Отладочный вывод
        # Проходим через полносвязный слой
        x = self.fc(x)
        return x


batch_size = 32

train_data = torch.load('data/train.pt')
# test_data = torch.load('data/test.pt')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

generator = Generator()
discriminator = Discriminator()

# Оптимизаторы (betas - пока чёрный ящик)
optimizer_gen = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_dis = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Функция потерь
criterion = nn.BCELoss()

num_epochs = 50

# Обучение
# for epoch in range(num_epochs):
#     return print('resr')

# Генерация шума для одного изображения (batch_size = 1)
noise = torch.randn(1, generator.noise_size)  # Тензор размерности (1, 100)

# Прямой проход через генератор
image_gen = generator(noise)
print(image_gen.shape)
Isdiscriminator = discriminator(image_gen)

print(Isdiscriminator.shape)
# print(image_gen.shape)  # Output: torch.Size([1, 3, 600, 400])

# # Загрузка модели
# model.load_state_dict(torch.load('models/autoencoder.pth'))









# Визуализация
def show_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)
    
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('output.png')  # Сохраняем изображение в файл
    plt.close()  # Закрываем фигуру

# show_image(fake_image[0])

# # Сохранение модели
# torch.save(model.state_dict(), 'autoencoder.pth')


