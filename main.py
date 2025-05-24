import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
from models.generator_rgb import GeneratorRGB
from models.generator_gray import GeneratorGray
import viewing
device = torch.device("cuda")
batch_size = 1

generator_gray = GeneratorGray().to(device)
generator_rgb = GeneratorRGB().to(device)

optimizer_g_gray = optim.Adam(generator_gray.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_g_rgb = optim.Adam(generator_rgb.parameters(), lr=0.0002, betas=(0.5, 0.999))

checkpoint = torch.load('models_save/checkpoint_gray.pth')
generator_gray.load_state_dict(checkpoint['generator_gray'])
optimizer_g_gray.load_state_dict(checkpoint['opt_gen_gray'])

checkpoint = torch.load('models_save/checkpoint_rgb.pth')
generator_rgb.load_state_dict(checkpoint['generator_rgb'])
optimizer_g_rgb.load_state_dict(checkpoint['opt_gen_rgb'])

for i in range(10):
    z = torch.randn(batch_size, 100, device=device)
    generat_gray = generator_gray(z)
    viewing.save_png_grey(generat_gray[0], f'{i}Gray')
    generat_rgb = generator_rgb(generat_gray)
    viewing.save_png(generat_rgb[0], f'{i}RGB')