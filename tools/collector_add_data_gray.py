import os
import numpy as np
from PIL import Image
import h5py

def add_gray_dataset(image_dir, output_file, img_size=(128, 128), batch_size=1000):
    # Получаем список изображений
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(image_dir)
        for file in files
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    # Открываем файл в режиме добавления ('a')
    with h5py.File(output_file, 'a') as hf:
        # Создаём датасет (если существует - будет ошибка, что нам и нужно)
        dset = hf.create_dataset(
            'faces_gray',
            shape=(0, 1, *img_size),
            maxshape=(None, 1, *img_size),
            dtype=np.float32,
            compression='gzip',
            chunks=(batch_size, 1, *img_size)
        )

        # Обрабатываем батчами
        for i in range(0, len(image_paths), batch_size):
            batch = []
            for path in image_paths[i:i + batch_size]:
                img = Image.open(path).convert('L').resize(img_size)
                img_array = (np.array(img) / 127.5) - 1.0
                img_array = np.expand_dims(img_array, axis=0)  # [1, H, W]
                batch.append(img_array)

            if batch:
                batch_arr = np.stack(batch)  # [N, 1, H, W]
                dset.resize((dset.shape[0] + batch_arr.shape[0], 1, *img_size))
                dset[-batch_arr.shape[0]:] = batch_arr

# Использование
add_gray_dataset(
    image_dir='../img/mix_img_128_gray',
    output_file='../data/faces.h5',
    img_size=(128, 128),
    batch_size=1000
)

with h5py.File('../data/faces.h5', 'r') as f:
    dset = f['faces_gray']
    print(dset.shape)