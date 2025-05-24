import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд

class DiscriminatorRGB(nn.Module):
    def __init__(self):
        super(DiscriminatorRGB, self).__init__()

        self.conv_blocks = nn.Sequential(
            # Блок 1: 128x128 -> 64x64 (3->64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 2: 64x64 -> 32x32 (64->128)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 3: 32x32 -> 16x16 (128->256)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Блок 4: 16x16 -> 8x8 (256->512)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [batch, 256, 1, 1]
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        return self.classifier(features)
