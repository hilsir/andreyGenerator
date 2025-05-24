import numpy as np
from perlin_noise import PerlinNoise
from PIL import Image
import random

output_path = '../img/img_noise/'

def gen_noise():
    # Размеры изображения
    width = 400
    height = 600

    # Генерация случайных значений для контраста и масштаба
    octaves = random.randint(2, 6)  # Случайное количество октав от 2 до 6
    # noise_scale = random.uniform(0.1, 0.9)  # Случайный масштаб шума
    contrast_factor = random.uniform(0.1, 3.0)  # Случайный контраст (чем выше — тем контрастнее)
    brightness_factor = random.uniform(0.1, 1.5)  # Случайная яркость (уменьшаем или увеличиваем яркость)

    # Случайный сдвиг для каждого канала
    noise_shift_r = random.uniform(0, 1)
    noise_shift_g = random.uniform(0, 1)
    noise_shift_b = random.uniform(0, 1)

    # Создаем объект PerlinNoise с случайным количеством октав
    noise = PerlinNoise(octaves=octaves, seed=random.randint(0, 1000))  # Случайное начальное значение (seed)

    # Генерация изображения
    image_data = np.zeros((height, width, 3), dtype=np.uint8)  # 3 канала (RGB)

    for i in range(height):
        for j in range(width):
            # Генерация шума для каждого пикселя с учетом случайных сдвигов
            value_r = noise([i / height + noise_shift_r, j / width])
            value_g = noise([i / height + noise_shift_g, j / width])
            value_b = noise([i / height + noise_shift_b, j / width])

            # Применяем масштаб и контраст
            red = int(((value_r + 1) * 127) * contrast_factor * brightness_factor)
            green = int(((value_g + 1) * 85) * contrast_factor * brightness_factor)
            blue = int(((value_b + 1) * 63) * contrast_factor * brightness_factor)

            # Ограничиваем значения в допустимый диапазон (0-255)
            red = min(max(red, 0), 255)
            green = min(max(green, 0), 255)
            blue = min(max(blue, 0), 255)

            # Заполняем пиксель цветами
            image_data[i, j] = [red, green, blue]

    # Создаем изображение из массива
    return Image.fromarray(image_data)

for i in range(1099, 1101):
    image = gen_noise()
    # Сохраняем изображение
    image.save(output_path + "image_" + str(i) + ".png")
    print(i)