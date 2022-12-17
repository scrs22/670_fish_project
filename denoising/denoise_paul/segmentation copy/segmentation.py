import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import scipy.ndimage



def GaussianBlur(i):
    return cv2.GaussianBlur(i, (5,5), 0)

def MedianBlur(i, winsize = 5):
    return cv2.medianBlur(i, winsize)

def OtsuThreshold(i):
    t, im = cv2.threshold((i*255).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return t, im

def Dilation(i, winsize=5):
    kernal = np.ones((winsize,winsize), np.uint8)
    erosion = cv2.erode(i,kernal,iterations = 1)
    return erosion

def Erosion(i, winsize=5):
    kernal = np.ones((winsize,winsize), np.uint8)
    dilation = cv2.dilate(i,kernal,iterations = 1)
    return dilation

#Closing is reverse of Opening, Dilation followed by Erosion. 
# It is useful in closing small holes inside the foreground objects, 
# or small black points on the object.
def MorphClosing(i, winsize=5):
    kernal = np.ones((winsize,winsize), np.uint8)
    image = cv2.morphologyEx(i, cv2.MORPH_CLOSE, kernal)
    return image

#Opening is just another name of erosion followed by dilation. 
# It is useful in removing noise, as we explained above.
def MorphOpening(i, winsize=5):
    kernal = np.ones((winsize,winsize), np.uint8)
    opening = cv2.morphologyEx(i, cv2.MORPH_OPEN, kernal)
    return opening    

def old_paper(image):
    image = np.uint8(image*255)
    t, image = OtsuThreshold(image)
    image = Dilation(image)
    image = Erosion(image)
    return image

def scallop(image):
    image = np.uint8(image*255)
    # threshold,mask = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    avg = np.mean(image)
    glare_threshold = avg*1.4
    # image = image*(image<glare_threshold) + avg*(image)
    mask = np.uint8(image>glare_threshold)
    image = cv2.inpaint(image,mask,5,cv2.INPAINT_TELEA)
    image = GaussianBlur(image)
    return image

def flounder(image):
    # # global
    image = np.uint8(image*255)
    # threshold,mask = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    avg = np.mean(image)
    glare_threshold = avg*1.4
    # image = image*(image<glare_threshold) + avg*(image)
    mask = np.uint8(image>glare_threshold)
    image = cv2.inpaint(image,mask,5,cv2.INPAINT_TELEA)
    # t, image = OtsuThreshold(image)
    image = GaussianBlur(image)
    # image = MorphOpening(image)
    
    # t, image = OtsuThreshold(image)
    # image = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    # return image
    # image = image*(mask==0) + (mask>0)*avg
    # image = image/225
    # image = MedianBlur(image)
    # image = Dilation(image)
    # image = Erosion(image)
    
    # online
    # image = np.uint8(GaussianBlur(image)*255)
    # threshold,output = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # # find the nearest spot to get value
    # image = cv2.inpaint(image,mask,10,cv2.INPAINT_NS)
    # image = image/225

    # image = scipy.ndimage.filters.sobel(image)
    # mask = Dilation(mask)
    
    
    # image = Dilation(image)
    
    return image
    
# read the image
show = True
fishType = 'flounder'
folder = 'train'

cur_path = os.getcwd()
imageDir = os.path.join(cur_path, "..",'dataset_cropped_split_blurScore',folder,fishType,'images')
storeDir = os.path.join(cur_path, "..",'dataset_cropped_denoise',folder,fishType,'images')
ds = '.DS_Store'

count = 0
for img_name in os.listdir(imageDir):
    if img_name == ds:
        continue
    
    print(img_name)
    # if img_name != '26.png':
    #     continue
    
    img = cv2.imread(os.path.join(imageDir, img_name))
    img = np.float32(rgb2gray(img))
    image = flounder(img)
    
    
    if show:
        plt.figure(count)
        plt.axis('off')

        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title('Input')

        plt.subplot(122)
        plt.imshow(image, cmap='gray')
        plt.title('Denoise Image')
        
        plt.tight_layout()
        
        plt.show()
    else:
        # save the images
        plt.savefig('foo.png')
        
    
    count += 1
    if count == 10:
        break
    

# img = cv2.imread('7.png')
# GrayScale
# img = np.float32(rgb2gray(img))

# image = flounder(img)


