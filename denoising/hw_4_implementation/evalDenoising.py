import os
import time

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image

from dip import EncDec
from utils import imread
from utils import gaussian
from scipy import ndimage
from scipy import signal


image_name = 'herring_31.jpg'
dpi = 300

noise1 = imread(f'./source/{image_name}')
noise1 = np.dot(noise1[...,:3], [0.2989, 0.5870, 0.1140])

################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################

# Create network
net = EncDec()

# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size()).detach()

###
# Your training code goes here.
###

loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

n_epoch = 500

training_errors = np.zeros(n_epoch)
for batch_i in range(n_epoch):
    if batch_i % 100 == 0:
        print(f"Finished {batch_i} epochs.")
    optimizer.zero_grad()  # Zero out old gradient

    # Forward and backward pass
    predicted_img = net(eta)
    train_loss = loss_fn(predicted_img, noisy_img)
    training_errors[batch_i] = train_loss

    train_loss.backward()
    optimizer.step()

out = net(eta)  # eta: (1, 1, 384, 256)
out_img = out[0, 0, :, :].transpose(0, 1).detach().numpy()


plt.figure(figsize=(1500/dpi, 400/dpi), dpi=dpi)
plt.title("Denoising Method Comparison")
plt.axis('off')
plt.tight_layout()

# Noisy version
plt.subplot(152)
plt.axis('off')
plt.imshow(noise1, cmap='gray')
plt.title(f'Original {image_name}', fontsize=6)

# Apply Gaussian filter
plt.subplot(153)
plt.axis('off')
gaussian_filter = gaussian(7, 2)
im_gauss = ndimage.convolve(noise1, gaussian_filter, mode="nearest")
plt.imshow(im_gauss, cmap='gray')
plt.title(f'Gaussian {image_name}', fontsize=6)

# Apply Median filter
plt.subplot(154)
plt.axis('off')
im_med = signal.medfilt(noise1, 7)
plt.imshow(im_med, cmap='gray')
plt.title(f'Median {image_name}', fontsize=6)

plt.subplot(155)
plt.axis('off')
plt.imshow(out_img, cmap='gray')
plt.title(f'Deep Img Prior {image_name}', fontsize=6)

plt.savefig(
    f'/Users/Cody/Documents/Classes/cs670/Project/project_github_repo/denoising/hw_4_implementation/img/comparison/{image_name}-{n_epoch}.png',
    dpi=500
)
plt.show()

img = Image.fromarray(out_img).convert('RGB')
img.save(f'/Users/Cody/Documents/Classes/cs670/Project/project_github_repo/denoising/hw_4_implementation/img/{image_name}-{n_epoch}-nn.png',)
img = Image.fromarray(im_gauss).convert('RGB')
img.save(f'/Users/Cody/Documents/Classes/cs670/Project/project_github_repo/denoising/hw_4_implementation/img/{image_name}-{n_epoch}-gaus.png',)
img = Image.fromarray(im_med).convert('RGB')
img.save(f'/Users/Cody/Documents/Classes/cs670/Project/project_github_repo/denoising/hw_4_implementation/img/{image_name}-{n_epoch}-med.png',)


plt.close()
print("Finished Denoising.")
