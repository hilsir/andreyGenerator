import torch
import matplotlib.pyplot as plt
import torchvision
plt.switch_backend('agg')  # Это может понадобиться в некоторых случаях
from PIL import Image

def viewing_date(path):
    # Загружаем данные
    train_data = torch.load(path)

    # Достаем изображения и метки
    train_images = train_data[0]
    train_labels = train_data[1]

    torchvision.utils.save_image(train_images[:10], 'generated_images/test.png')

def save_png(img,epoch):
    # Подготовка к сохранению
    image = img.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    image = (image + 1) / 2 * 255  # [-1, 1] -> [0, 255]
    image = image.clamp(0, 255).byte().cpu().numpy()

    # Сохранение
    Image.fromarray(image).save(f"generated_images/image_{epoch}.png")

def save_png_grey(img,epoch):
    # Подготовка изображения
    image = img.squeeze(0).squeeze(0)  # Удаляем batch и channel (если 1 канал)
    image = (image + 1) / 2 * 255  # [-1, 1] -> [0, 255]
    image = image.clamp(0, 255).byte().cpu().numpy()

    # Сохранение как 8-bit grayscale
    Image.fromarray(image, mode='L').save(f"generated_images/image_{epoch}.png")

