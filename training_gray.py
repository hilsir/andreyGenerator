import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд


from models.generator_gray import GeneratorGray
from models.discriminator_gray import DiscriminatorGray
import viewing

device = torch.device("cuda")

batch_size = 32

# train_images = torch.load('data/andre_grey.pt')
train_images = torch.load('data/faces_gray.pt')
# Создаём DataLoader (нету меток)
train_loader = DataLoader(
    TensorDataset(train_images),
    batch_size=32,
    shuffle=True,
    drop_last=True
)

generator_gray = GeneratorGray().to(device)
discriminator_gray = DiscriminatorGray().to(device)

# start_epoch = 0
num_epochs = 100000

optimizer_g_gray = optim.Adam(generator_gray.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d_gray = optim.Adam(discriminator_gray.parameters(), lr=0.00002, betas=(0.5, 0.999))

# Комплексная загрузка
checkpoint = torch.load('models_save/checkpoint_gray.pth')
generator_gray.load_state_dict(checkpoint['generator_gray'])
discriminator_gray.load_state_dict(checkpoint['discriminator_gray'])
optimizer_g_gray.load_state_dict(checkpoint['opt_gen_gray'])
optimizer_d_gray.load_state_dict(checkpoint['opt_dis_gray'])
start_epoch = checkpoint.get('epoch', 0) + 1  # Продолжаем со следующей эпохи

criterion = nn.BCELoss().to(device)

for epoch in range(start_epoch, num_epochs):

    d_loss = torch.tensor(0.0, device=device)
    g_loss = torch.tensor(0.0, device=device)

    is_training_d = True

    # enumerate() добавляет счётчик к итерации
    for i, (real_imgs, ) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)

        # Подготовка данных
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Генерация фейковых изображений
        z = torch.randn(batch_size, 100, device=device)
        fake_imgs = generator_gray(z)

        # ======== Обучение дискриминатора ========
        if(is_training_d):
            optimizer_d_gray.zero_grad()
            # Ошибка на реальных изображениях
            real_output = discriminator_gray(real_imgs)
            real_loss = criterion(real_output, real_labels)

            # Ошибка на фейковых изображениях
            fake_output = discriminator_gray(fake_imgs.detach())  # detach чтобы не считать градиенты для генератора
            fake_loss = criterion(fake_output, fake_labels)

            d_loss = (real_loss + fake_loss) / 2  # Среднее двух ошибок
            d_loss.backward()
            optimizer_d_gray.step()

        # ======== Обучение генератора ========
        optimizer_g_gray.zero_grad()

        # Повторно генерируем изображения (или можно использовать те же fake_imgs)
        z = torch.randn(batch_size, 100, device=device)
        fake_imgs = generator_gray(z)

        # Ошибка генератора (пытаемся обмануть дискриминатор)
        g_output = discriminator_gray(fake_imgs)
        g_loss = criterion(g_output, real_labels)  # Цель - чтобы дискриминатор думал, что это реальные

        g_loss.backward()
        optimizer_g_gray.step()

        if (g_loss.item() > 1.5):
            is_training_d = False
        else:
            is_training_d = True



    # вывод изображения
    z = torch.randn(1, 100).to(device)
    viewing.save_png_grey(generator_gray(z), epoch)

    # После расчёта loss_gen и loss_dis
    if d_loss.item() != 0:
        ratio = g_loss.item() / d_loss.item()  # Соотношение ошибок
        print(
            f"E: {epoch} | G/D Баланс ошибок: {ratio:.2f} | G Loss: {g_loss.item():.8f} | D Loss: {d_loss.item():.8f}")
    else:
        print(
            f"E: {epoch} | G/D Баланс ошибок: - | G Loss: {g_loss.item():.8f} | D Loss: не обучался")

    if epoch % 1 == 0:
        print("save")
        checkpoint = {
            'epoch': epoch,
            'generator_gray': generator_gray.state_dict(),
            'discriminator_gray': discriminator_gray.state_dict(),
            'opt_gen_gray': optimizer_g_gray.state_dict(),
            'opt_dis_gray': optimizer_d_gray.state_dict(),
        }
        torch.save(checkpoint, 'models_save/checkpoint_gray.pth')




























