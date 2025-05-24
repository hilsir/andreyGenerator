import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд

class DiscriminatorGray(nn.Module):
    def __init__(self):
        super(DiscriminatorGray, self).__init__()

        self.conv_blocks = nn.Sequential(
            # Блок 1: 128x128 -> 64x64
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Блок 2: 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Блок 3: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Блок 4: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Блок 5: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Блок 5: 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
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
