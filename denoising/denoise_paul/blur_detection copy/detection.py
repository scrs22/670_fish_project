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

clear_fileNames = [f for f in os.listdir(clear_dataDir) if os.path.isfile(os.path.join(clear_dataDir, f))]
blur_fileNames = [f for f in os.listdir(blur_dataDir) if os.path.isfile(os.path.join(blur_dataDir, f))]

clear_scores = []
blur_scores = []

for clear_fileName in clear_fileNames:
    if clear_fileName == '.DS_Store':
        continue
    # if clear_fileName != '0_IPHONE-SE_S.JPG':
    #     continue
    img_path = os.path.join(clear_dataDir, clear_fileName)
    img = imread(img_path)
    gray_img = rgb2gray(img)
    clear_scores.append(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    # print(clear_fileName)
    # break


clear_scores.sort()

for blur_fileName in blur_fileNames:
    if blur_fileName == '.DS_Store':
        continue
    # if blur_fileName != '0_IPHONE-SE_F.JPG':
    #     continue
    img_path = os.path.join(blur_dataDir, blur_fileName)
    img = imread(img_path)
    gray_img = rgb2gray(img)
    blur_scores.append(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    # print(blur_fileName)
    # break

# plt.hist(blur_scores)
# plt.show()
blur_scores.sort()
# print(f"blur: {blur_scores}")
# print(f"clear: {clear_scores}")

threshold = 0
accuracy = 0
# clear image has larger score, blur has lower score
for i in range(len(clear_scores)):
    clear = clear_scores[i]
    for j in range(1,len(blur_scores)+1):
        blur = blur_scores[-j]
        if clear < blur:
            continue
        else:
            break
    temp_acc = (len(clear_scores) - i + len(blur_scores) - j) / (len(clear_scores) + len(blur_scores))
    if temp_acc > accuracy:
        accuracy = temp_acc
        threshold = clear

print(f'Threshold: {threshold}, Accuracy: {accuracy}')
# This file train on the blur dataset we found, goal find the best threshold for blur score
# Threshold: 0.002946554451218414, Accuracy: 0.8161904761904762

