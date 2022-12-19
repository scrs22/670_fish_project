import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import scipy.ndimage
from PIL import Image  

def GaussianBlur(i):
    return cv2.GaussianBlur(i, (5,5), 0)

def MedianBlur(i, winsize = 5):
    return cv2.medianBlur(i, winsize)

#Opening is just another name of erosion followed by dilation. 
# It is useful in removing noise, as we explained above.
def MorphOpening(i, winsize=5):
    kernal = np.ones((winsize,winsize), np.uint8)
    opening = cv2.morphologyEx(i, cv2.MORPH_OPEN, kernal)
    return opening   

def denoise(image):
    image = np.uint8(GaussianBlur(image)*255)
    avg = np.mean(image)
    glare_threshold = avg*1.4
    mask = np.uint8(image>glare_threshold)
    image = cv2.inpaint(image,mask,5,cv2.INPAINT_TELEA)
    image = GaussianBlur(image)
    image = MorphOpening(image)
    
    return image

def write2folder():
    fishTypes = ['dead-scallop','flounder','herring','roundfish','scallop','skate']
    folders = ['train', 'test', 'val']
    cur_path = os.getcwd()
    ds = '.DS_Store'


    for folder in folders:
        for fishType in fishTypes:
            imageDir = os.path.join(cur_path, "..",'dataset_cropped_split_blurScore',folder,fishType,'images')
            storeDir = os.path.join(cur_path, "..",'dataset_cropped_denoise',folder,fishType,'images')
            for img_name in os.listdir(imageDir):
                if img_name == ds:
                    continue
                img = cv2.imread(os.path.join(imageDir, img_name))
                img = np.float32(rgb2gray(img))
                image = denoise(img)
                im = Image.fromarray(image)
                im.save(os.path.join(storeDir,img_name))
            print(f'Done: {folder}, {fishType}')
                
def checkData():
    fishTypes = ['dead-scallop','flounder','herring','roundfish','scallop','skate']
    folders = ['train', 'test', 'val']
    cur_path = os.getcwd()
    ds = '.DS_Store'

    count = 0
    denoise_count = 0
    for folder in folders:
        for fishType in fishTypes:
            imageDir = os.path.join(cur_path, "..",'dataset_cropped_split_blurScore',folder,fishType,'images')
            storeDir = os.path.join(cur_path, "..",'dataset_cropped_denoise',folder,fishType)
            for img_name in os.listdir(imageDir):
                if img_name == ds:
                    continue
                file_name = os.path.splitext(img_name)[0]
                count += 1
                denoise_count += 1
                if not os.path.exists(os.path.join(storeDir,'images',img_name)):
                    print(f'{folder},{fishType},{img_name}: image not exist')
                    
                if not os.path.exists(os.path.join(storeDir,'blur',file_name+'.txt')):
                    print(os.path.join(storeDir,'blur',file_name+'.txt'))
                    
                if not os.path.exists(os.path.join(storeDir,'labels',file_name+'.txt')):
                    print(f'{folder},{fishType},{img_name}: labels not exist')
    
    # After all
    print(f'There are total of {count} number of images in original dataset')
    print(f'There are total of {denoise_count} number of images in denoise dataset')


checkData()
                    