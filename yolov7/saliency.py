import math

import cv2
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmyolo.utils import register_all_modules
from mmengine.registry import DefaultScope

def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask

def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked

def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)

def generate_saliency_map(image,
                          target_box,
                          prob_thresh=0.5,
                          grid_size=(16, 16),
                          n_masks=5000,
                          seed=0):
    np.random.seed(seed)
    image_h, image_w = image.shape[:2]
    res = np.zeros((image_h, image_w), dtype=np.float32)
    for _ in range(n_masks):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)
        masked = mask_image(image, mask)
        out = inference_detector(model, masked)
        preds=out.pred_instances
        bboxes,scores=preds.bboxes.cpu().data.numpy(),preds.scores.cpu().data.numpy()
        temp=[]
        for i, bbox in enumerate(bboxes):
            if scores[i] < 0.2577:
                break
            box = tuple(np.round(bbox).astype(int).tolist())
            temp.append(iou(target_box, box) * scores[i])
        max_score = max(temp,default=0)
        res += mask * max_score
    return res


image_path = "data/input/test/herring/images/319.png"
config = 'config.py'
checkpoint ='runs/train/uncropped/weights/best.pt'
device = 'cuda:0'

register_all_modules()

model = init_detector(config, checkpoint, device)
label_names = ['scallop','herring','dead-scallop','flounder','roundfish','skate']
image = cv2.imread(image_path)
scale = 600 / min(image.shape[:2])
image = cv2.resize(image,
                  None,
                  fx=scale,
                  fy=scale,
                  interpolation=cv2.INTER_AREA)


out = inference_detector(model, image)
res = image.copy()
preds=out.pred_instances
bboxes,labels,scores=preds.bboxes.cpu().data.numpy(),preds.labels.cpu().data.numpy(),preds.scores.cpu().data.numpy()
for i, bbox in enumerate(bboxes):
    if scores[i] < 0.1:
        break
    box = tuple(np.round(bbox).astype(int).tolist())
    print(i, labels[i], box, scores[i])
    cv2.rectangle(res, box[:2], box[2:], (0, 255, 0), 5)
# plt.figure(figsize=(7, 7))
# plt.imshow(res[:, :, ::-1])
# plt.show()

target_box = np.array([225, 313, 472, 806])
saliency_map = generate_saliency_map(image,
                                     target_box=target_box,
                                     prob_thresh=0.5,
                                     grid_size=(16, 16),
                                     n_masks=1000)

image_with_bbox = image.copy()
saliency_map=saliency_map.astype(np.float32)
cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
              (0, 255, 0), 5)
plt.figure(figsize=(7, 7))
plt.imshow(image_with_bbox[:, :, ::-1])
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
print(saliency_map)
plt.imsave(saliency_map,'saliency.png',cmap='jet')
plt.show()