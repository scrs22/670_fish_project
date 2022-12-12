import sys
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
import math
from utils.datasets import letterbox

def visualise(weights_path,img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_names =[ 'scallop','herring','dead-scallop','flounder','roundfish','skate' ]
    weigths = torch.load(weights_path)
    model = weigths['model']
    model = model.half().to(device)
    _ = model.eval()
    image = cv2.imread(img_path)  # 504x378 image
    
    # imaget = letterbox(image, 640, stride=64, auto=True)[0]
    # image_t = image.copy()
    # imaget = transforms.ToTensor()(image)
    # imaget = torch.tensor(np.array([image.numpy()]))
    # imaget = image.to(device)
    # imaget = image.half()

    

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
        masked = letterbox(masked, 1280, stride=64, auto=True)[0]
        masked_ = image.copy()
        masked = transforms.ToTensor()(masked)
        masked = torch.tensor(np.array([masked.numpy()]))
        masked = masked.to(device)
        masked = masked.half()
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
                              target_class_index,
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
            out = model(masked)
            print(out)
            # sys.stdout.flush()
            # pred = out[target_class_index]
            # score = max([iou(target_box, box) * score for *box, score in pred],
            #             default=0)
            # res += mask * score
        return out

    target_box = np.array([289, 72, 491, 388])
    saliency_map = generate_saliency_map(image,
                                        target_class_index=15,
                                        target_box=target_box,
                                        prob_thresh=0.5,
                                        grid_size=(16, 16),
                                        n_masks=1000)
    return saliency_map
    
    # image_with_bbox = image.copy()
    # cv2.rectangle(image_with_bbox, tuple(target_box[:2]), tuple(target_box[2:]),
    #               (0, 255, 0), 5)
    # plt.figure(figsize=(7, 7))
    # plt.imshow(image_with_bbox[:, :, ::-1])
    # plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    # plt.axis('off')
    # plt.imsave("uncropped")
    # plt.show()

if __name__ == "__main__":
    image=visualise( 'runs/train/uncropped/weights/best.pt','data/input/test/herring/images/319.png')
    print(image)




