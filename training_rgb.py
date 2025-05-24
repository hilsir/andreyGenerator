import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
import h5py
from models.generator_rgb import GeneratorRGB
from models.discriminator_rgb import DiscriminatorRGB
import viewing

device = torch.device("cuda")
file_path_faces = 'data/faces.h5'
batch_size = 64
# start_epoch = 0
num_epochs = 100


generator_rgb = GeneratorRGB().to(device)
discriminator_rgb = DiscriminatorRGB().to(device)
optimizer_g_rgb = optim.Adam(generator_rgb.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_d_rgb = optim.Adam(discriminator_rgb.parameters(), lr=0.00002, betas=(0.5, 0.999))
lossL1 = nn.L1Loss().to(device)
lossMSE = nn.MSELoss().to(device)
# Метки
real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)

# Комплексная загрузка
checkpoint = torch.load('models_save/checkpoint_rgb.pth')
generator_rgb.load_state_dict(checkpoint['generator_rgb'])
optimizer_g_rgb.load_state_dict(checkpoint['opt_gen_rgb'])
start_epoch = checkpoint.get('epoch', 0) + 1  # Продолжаем со следующей эпохи

with h5py.File(file_path_faces, 'r') as f:
    dset_rgb = f['faces_rgb']
    dset_gray = f['faces_gray']
    gray_dataset_size = dset_gray.shape[0]

    for epoch in range(start_epoch, num_epochs):
        # Проходимся по батчами
        for batch_iteration, batch_start in enumerate(range(0, gray_dataset_size, batch_size)):
            # Получаем текущий батч
            batch_end = batch_start + batch_size

            # Пропускаем последний неполный батч
            if batch_end > gray_dataset_size:
                print(f"Пропуск неполного батча {batch_start}-{batch_end} (размер датасета: {gray_dataset_size})")
                continue

            # батчи на итерацию
            batch_gray = dset_gray[batch_start:batch_start + batch_size]
            batch_rgb = dset_rgb[batch_start:batch_start + batch_size]
            # NumPy -> PyTorch-тензор
            batch_tensor_gray = torch.from_numpy(batch_gray).to(device)
            batch_tensor_rgb = torch.from_numpy(batch_rgb).to(device)

            # ======== Обучение генератора ========
            optimizer_g_rgb.zero_grad()
            fake_imgs = generator_rgb(batch_tensor_gray)
            g_lossL1 = lossL1(fake_imgs, batch_tensor_rgb)
            g_lossMSE = lossMSE(fake_imgs, batch_tensor_rgb)
            g_loss = g_lossL1 + g_lossMSE
            g_loss.backward()
            optimizer_g_rgb.step()

            print(
                f"batch {batch_iteration} | G Loss: {g_loss.item():.8f}")

            if batch_iteration % 100 == 0 and batch_iteration != 0:
                viewing.save_png(fake_imgs[0], f'{batch_iteration}F')
                viewing.save_png(batch_tensor_rgb[0], f'{batch_iteration}R')
                print("save")
                checkpoint = {
                    'epoch': epoch,
                    'generator_rgb': generator_rgb.state_dict(),
                    'opt_gen_rgb': optimizer_g_rgb.state_dict(),
                }
                torch.save(checkpoint, 'models_save/checkpoint_rgb.pth')


        # one_img_rgb = batch_tensor_rgb[0].unsqueeze(0)  # Берем первое изображение
        # print(one_img_rgb.shape)
        # viewing.save_png(one_img_rgb, 1)



