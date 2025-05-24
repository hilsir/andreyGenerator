import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд

class GeneratorGray(nn.Module):
    def __init__(self):
        super(GeneratorGray, self).__init__()

        self.noise_size = 100  # Размерность входного шума
        self.img_height = 128
        self.img_width = 128
        # Используем 4 увеличения: 8->16->32->64->128
        self.init_height = 8
        self.init_width = 8

        # Fully Connected Layer
        self.fc_noise = nn.Sequential(
            nn.Linear(self.noise_size, 512 * self.init_height * self.init_width),
            nn.LayerNorm(512 * self.init_height * self.init_width),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Сверточные блоки
        self.conv_blocks = nn.Sequential(
            # Блок 1: 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 2: 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 3: 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 4: 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Финальный слой
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        x = self.fc_noise(noise)
        x = x.view(-1, 512, self.init_height, self.init_width)
        img = self.conv_blocks(x)
        return img
