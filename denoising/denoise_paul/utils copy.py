import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

def imread(path):
    img = plt.imread(path).astype(float)
    
    #Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    #Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0

    return img

def blurScore(img):
    gray_img = rgb2gray(img)
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()