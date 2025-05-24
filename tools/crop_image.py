import os
from PIL import Image

# Путь к папке с изображениями
input_folder = 'img/img_andre/class_1'
output_folder = 'img/img_andre/class_3'  # Папка для сохранения обрезанных изображений

# Создаем папку для результатов, если её нет
os.makedirs(output_folder, exist_ok=True)

# Желаемые размеры
target_width = 400
target_height = 512

# Проходим по всем файлам в папке
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        # Открываем изображение
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Проверяем размеры исходного изображения
        width, height = img.size
        if width != 400 or height != 600:
            print(f"Предупреждение: {filename} имеет размер {width}x{height}, ожидается 400x600")
            continue

        # Вычисляем координаты для обрезки (по центру)
        left = 0
        top = (height - target_height) // 2  # Обрезаем равномерно сверху и снизу
        right = width
        bottom = top + target_height

        # Обрезаем изображение
        cropped_img = img.crop((left, top, right, bottom))

        # Сохраняем результат
        output_path = os.path.join(output_folder, filename)
        cropped_img.save(output_path)

        print(f"Обработано: {filename}")

print("Все изображения обрезаны и сохранены в", output_folder)