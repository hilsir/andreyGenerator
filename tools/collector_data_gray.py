import os
import numpy as np
from PIL import Image
import h5py

def create_hdf5_dataset(image_dir, output_file, img_size=(128, 128), batch_size=10000):
    # Получаем список всех изображений
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    # Создаем HDF5 файл 'w' создаёт 'a' добавляет
    with h5py.File(output_file, 'w') as hf:
        # Создаем расширяемый датасет для одноканальных изображений
        dset = hf.create_dataset(
            'faces_gray',
            shape=(0, 1, *img_size),  # 1 канал вместо 3
            maxshape=(None, 1, *img_size),
            dtype=np.float32,
            compression='gzip',
            chunks=(batch_size, 1, *img_size)
        )

        # Обрабатываем изображения батчами
        for i in range(0, len(image_paths), batch_size):
            batch = []
            for path in image_paths[i:i + batch_size]:
                img = Image.open(path).convert('L')  # 'L' - режим для grayscale
                img = img.resize(img_size)
                img_array = (np.array(img) / 127.5) - 1.0  # Нормализация [-1, 1]
                img_array = np.expand_dims(img_array, axis=0)  # Добавляем ось канала -> [1, H, W]
                batch.append(img_array)

            if batch:
                batch_arr = np.stack(batch)  # [N, 1, H, W]
                dset.resize((dset.shape[0] + batch_arr.shape[0], 1, *img_size))
                dset[-batch_arr.shape[0]:] = batch_arr

create_hdf5_dataset(
    image_dir='../img/test_grey',
    output_file='../data/faces.h5',
    img_size=(128, 128),
    batch_size=1000
)