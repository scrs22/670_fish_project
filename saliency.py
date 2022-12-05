
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector


config = 'yolov7/data/herring.yaml'
checkpoint = 'yolov7/runs/train/yolov7-custom16/weights/best.pt'
device = 'cuda:0'
model = init_detector(config, checkpoint, device)
label_names = [ 'scallop','herring','dead-scallop','flounder','roundfish','skate' ]


image_path = "/Users/schanumolu/Desktop/670/project/670_fish_project/yolov7/data/input/test/flounder/images/13.png"
image = cv2.imread(image_path)
scale = 600 / min(image.shape[:2])
image = cv2.resize(image,
                   None,
                   fx=scale,
                   fy=scale,
                   interpolation=cv2.INTER_AREA)
plt.figure(figsize=(7, 7))
plt.imshow(image[:, :, ::-1])
plt.show()


