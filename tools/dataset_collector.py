import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

# Путь к папке с изображениями

data_dir = '../img/img_andre'

def collector_data_rgb():
    # Преобразования для изображений
    transform = transforms.Compose([
        transforms.ToTensor(),  # Конвертирует в тензор и нормализует в [0, 1]
    ])

    # Загрузка всего датасета
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Разделение на train и test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Функция для сохранения в виде кортежа (images, labels)
    def save_as_tuple(dataset, filename):
        """Сохраняет данные как кортеж (тензор_изображений, тензор_меток)"""
        images = torch.stack([x[0] for x in dataset])  # shape: [N, C, H, W]
        labels = torch.tensor([x[1] for x in dataset])  # shape: [N]
        torch.save((images, labels), filename)  # Сохраняем кортеж

    # Сохраняем datasets
    save_as_tuple(train_dataset, 'data/train_rgb.pt')
    save_as_tuple(test_dataset, 'data/test_rgb.pt')


def collector_data_gray():
    # Преобразования для изображений (теперь с конвертацией в grayscale)
    transform = transforms.Compose([
        transforms.Grayscale(),  # Конвертирует в оттенки серого
        transforms.ToTensor(),   # Конвертирует в тензор и нормализует в [0, 1]
    ])

    # Загрузка всего датасета
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Разделение на train и test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Функция для сохранения в виде кортежа (images, labels)
    def save_as_tuple(dataset, filename):
        """Сохраняет данные как кортеж (тензор_изображений, тензор_меток)"""
        images = torch.stack([x[0] for x in dataset])  # shape: [N, 1, H, W] (1 канал для grayscale)
        labels = torch.tensor([x[1] for x in dataset])  # shape: [N]
        torch.save((images, labels), filename)  # Сохраняем кортеж

    # Сохраняем datasets (измените имена файлов на 'gray' вместо 'rgb')
    save_as_tuple(train_dataset, 'data/train_gray.pt')
    save_as_tuple(test_dataset, 'data/test_gray.pt')

collector_data_gray()
collector_data_rgb()