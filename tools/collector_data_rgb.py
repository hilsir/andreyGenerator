import torch
from torchvision import transforms
from PIL import Image
import os
from multiprocessing import Pool
import numpy as np
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
        # Создаем расширяемый датасет
        dset = hf.create_dataset(
            'faces_rgb',
            shape=(0, 3, *img_size),
            maxshape=(None, 3, *img_size),
            dtype=np.float32,
            compression='gzip',
            chunks=(batch_size, 3, *img_size)
        )

        # Обрабатываем изображения батчами
        for i in range(0, len(image_paths), batch_size):
            batch = []
            for path in image_paths[i:i + batch_size]:
                img = Image.open(path).convert('RGB').resize(img_size)
                img_array = (np.array(img)/ 127.5) - 1.0 # лучшая нормализация для изображения из худших . . .
                img_array = np.transpose(img_array, (2, 0, 1))  # -> [3, H, W]
                batch.append(img_array)

            if batch:
                batch_arr = np.stack(batch)  # [N, 3, H, W]
                dset.resize((dset.shape[0] + batch_arr.shape[0], 3, *img_size))
                dset[-batch_arr.shape[0]:] = batch_arr

create_hdf5_dataset(
    image_dir='../img/mix_img_128',
    output_file='../data/faces.h5',
    img_size=(128, 128),
    batch_size=1000
)