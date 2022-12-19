import sys
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
import tqdm
import math
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

def visualise(weights_path,img_path,target_box,target_class_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes=np.array([ 'scallop','herring','dead-scallop','flounder','roundfish','skate' ])
    weights = torch.load(weights_path)
    model = attempt_load(weights_path, map_location=device)
    stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # model = model.half().to(device)
    # _ = model.eval()
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
        masked0 = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
                  255).astype(np.uint8)
        masked = letterbox(masked0, 640, stride=stride)[0]

        # Convert
        masked = masked[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        masked = np.ascontiguousarray(masked)
        masked = torch.from_numpy(masked).to(device)
        masked = masked.float()  # uint8 to fp16/32
        masked /= 255.0  # 0 - 255 to 0.0 - 1.0
        # print(masked)
        if masked.ndimension() == 3:
            masked = masked.unsqueeze(0)
        return masked0,masked

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
        max_score=0
        for _ in range(n_masks):
            mask = generate_mask(image_size=(image_w, image_h),
                                grid_size=grid_size,
                                prob_thresh=prob_thresh)
            masked0,masked = mask_image(image, mask)
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(masked)[0]
                # print(pred)
                pred = non_max_suppression(pred)
                # score = max([iou(target_box, box) * score for *box, score in pred],default=0)
                # res += mask * score
            max_score=0
            for i, det in enumerate(pred):
                # print(det[:, :4])
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(masked.shape[2:], det[:, :4], masked0.shape).round()
                    bboxes=det[:, :4].cpu().data.numpy()
                    scores=det[:, 4].cpu().data.numpy()
                    classes_=det[:, -1].cpu().data.numpy()
                    # print(bbox,score,class_)
                    for item in range(len(scores)):
                        if int(classes_[item])==int(target_class_index):
                            max_score=max(iou(target_box,bboxes[item])*scores[item],max_score)
            res += mask * max_score
        return res

    # target_box = np.array([270, 255, 615, 377])
    saliency_map = generate_saliency_map(image,
                                        target_class_index=target_class_index,
                                        target_box=target_box,
                                        prob_thresh=0.5,
                                        grid_size=(16, 16),
                                        n_masks=2000)
    image_with_bbox = image.copy()
    # print(image_with_bbox.shape)
    
    return image_with_bbox , saliency_map
    


if __name__ == "__main__":
    fish_images={'scallop':['1030',0],'herring':['65',1],'flounder':['507',3],'roundfish':['196',4],'skate':['672',5]}
    fish_target_boxes=[[510, 510, 670, 640],[260, 255, 615, 377],[286, 25, 650, 490],[620, 503, 928, 910],[328, 170, 700, 809]]
    fig, axes = plt.subplots(5, 2, figsize=(20,45))
    plt.subplots_adjust(wspace=0, hspace=0,left=0, right=1, bottom=0, top=1)
    plt.rcParams.update({'font.size': 48})
    count=0
    for fish in fish_images:
        # if count==2:
        #     count+=1
        #     continue
        ax1, ax2= axes[count]
        image_with_bbox,saliency_map=visualise( 'runs/train/uncropped/weights/best.pt',f'data/input/test/{fish}/images/{fish_images[fish][0]}.png',fish_target_boxes[count],fish_images[fish][1])
        cv2.rectangle(image_with_bbox, tuple(fish_target_boxes[count][:2]), tuple(fish_target_boxes[count][2:]),(0, 255, 0), 5)
        ax1.imshow(image_with_bbox[:, :, ::-1])
        ax1.axis('off') # for removing axis
        ax2.axis('off')
        ax1.set_title(f'{fish}')
        ax2.imshow(saliency_map, cmap='jet', alpha=0.5)
        ax2.set_title('Saliency Map')
        count+=1
    plt.savefig("uncropped.png")

    # fish_images={'scallop':['1030',0],'herring':['65',1],'flounder':['507',3],'roundfish':['196',4],'skate':['672',5]}
    # fish_target_boxes=[260, 255, 615, 377]
    # fig, axes = plt.subplots(1, 4, figsize=(50,20))
    # plt.subplots_adjust(wspace=0, hspace=0,left=0, right=1, bottom=0, top=1)
    # plt.rcParams.update({'font.size': 50})
    # count=0
    # den=['uncropped','denoise8','nafnet3']
   
    # if count==2:
    #     count+=1
    #     continue
    # ax1, ax2,ax3,ax4= axes
    # image_with_bbox,saliency_map_1=visualise( f'runs/train/uncropped/weights/best.pt',f'data/input/test/herring/images/65.png',fish_target_boxes,1)
    # cv2.rectangle(image_with_bbox, tuple(fish_target_boxes[:2]), tuple(fish_target_boxes[2:]),(0, 255, 0), 5)
    # ax1.imshow(image_with_bbox[:, :, ::-1])
    # ax1.axis('off') # for removing axis
    # ax2.axis('off')
    # ax3.axis('off')
    # ax4.axis('off')
    # ax1.set_title('Herring fish')
    # ax2.imshow(saliency_map_1, cmap='jet', alpha=0.5)
    # ax2.set_title('No enhancement')
    # _,saliency_map_2=visualise( f'runs/train/denoise8/weights/best.pt',f'data/input_nf/test/herring/images/65.png',fish_target_boxes,1)
    # ax3.imshow(saliency_map_2, cmap='jet', alpha=0.5)
    # ax3.set_title('Feature enhanced')
    # _,saliency_map_3=visualise( f'runs/train/nafnet3/weights/best.pt',f'data/input_nf/test/herring/images/65.png',fish_target_boxes,1)
    # ax4.imshow(saliency_map_3, cmap='jet', alpha=0.5)
    # ax4.set_title('Model enhanced')
    # plt.savefig("saliency_enhancement.png")
    
    
    
    

    


