import numpy as np
from skimage.color import rgb2gray
import cv2
import os
import matplotlib.pyplot as plt

clear_dataDir = os.path.join('data', 'clear')
blur_dataDir = os.path.join('data', 'blur')

def imread(path):
    img = plt.imread(path).astype(float)
    
    #Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    #Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0

    return img

clearImg = imread('deblur_img.png')
blurImg = imread('blurry.jpg')

clear = rgb2gray(clearImg)
blur = rgb2gray(blurImg)

clear_score = cv2.Laplacian(clear, cv2.CV_64F).var()
blur_score = cv2.Laplacian(blur, cv2.CV_64F).var()

print(f'clear: {clear_score}')
print(f'blur: {blur_score}')