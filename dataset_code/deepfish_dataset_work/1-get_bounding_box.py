import os
import shutil
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

ds_images_valid = Path('deepfish_ds/images/valid')
ds_images_empty = Path('deepfish_ds/images/empty')
ds_images_masks = Path('deepfish_ds/masks/valid')

target_dir_img = Path('dataset_gs/all/images')
target_dir_text = Path('dataset_gs/all/labels')
target_dir_boxed = Path('dataset_gs/all/boxed')

os.makedirs(target_dir_img, exist_ok=True)
os.makedirs(target_dir_text, exist_ok=True)
os.makedirs(target_dir_boxed, exist_ok=True)

file_idx = 0


# From homework utils
def imread(path):
    loaded_img = plt.imread(path).astype(float)
    # Remove alpha channel if it exists
    if loaded_img.ndim > 2 and loaded_img.shape[2] == 4:
        loaded_img = loaded_img[:, :, 0:3]
    # Puts images values in range [0,1]
    if loaded_img.max() > 1.0:
        loaded_img /= 255.0
    return loaded_img


def create_bounding_boxes(image: np.ndarray, class_name):
    seen = np.zeros_like(image)

    def _bfs(start_x, start_y):
        initial_values = [1e6, -1, 1e6, -1]
        min_x, max_x, min_y, max_y = initial_values
        neighbors = [(start_x, start_y)]
        while neighbors:
            curr_x, curr_y = neighbors.pop()
            if image[curr_y, curr_x] and not seen[curr_y, curr_x] > 0:
                if curr_x < min_x:
                    min_x = curr_x
                if curr_x > max_x:
                    max_x = curr_x
                if curr_y < min_y:
                    min_y = curr_y
                if curr_y > max_y:
                    max_y = curr_y

                next_coordinates = (
                    (curr_x - 1, curr_y - 1), (curr_x, curr_y - 1), (curr_x + 1, curr_y - 1),
                    (curr_x - 1, curr_y), (curr_x + 1, curr_y),
                    (curr_x - 1, curr_y + 1), (curr_x, curr_y + 1), (curr_x + 1, curr_y + 1),
                )

                for prospective_x, prospective_y in next_coordinates:
                    if 0 <= prospective_x < image.shape[1] and 0 <= prospective_y < image.shape[0]:
                        neighbors.append((prospective_x, prospective_y))
            seen[curr_y, curr_x] = 1

        if (
                min_x == initial_values[0] and
                max_x == initial_values[1] and
                min_y == initial_values[2] and
                max_y == initial_values[3]
        ):
            return None
        return min_x, max_x, min_y, max_y

    all_boxes = []
    all_boxes_str = ''
    for y in np.arange(image.shape[0]):
        for x in np.arange(image.shape[1]):
            boxes = _bfs(x, y)
            if boxes:
                all_boxes.append(boxes)
                y_bound = image.shape[0]
                x_bound = image.shape[1]

                x_center = ((boxes[0] + boxes[1]) // 2) / x_bound
                y_center = ((boxes[2] + boxes[3]) // 2) / y_bound
                width_norm = (boxes[1] - boxes[0]) / x_bound
                height_norm = (boxes[3] - boxes[2]) / y_bound

                all_boxes_str += f"{class_name} {x_center} {y_center} {width_norm} {height_norm}\n"

    if len(all_boxes_str) > 0:
        all_boxes_str = all_boxes_str[:-1]
    return all_boxes, all_boxes_str


num_images = len([name for name in os.listdir(ds_images_valid) if os.path.isfile(os.path.join(ds_images_valid, name))])
source_dir = ds_images_valid
for img_file in source_dir.iterdir():
    if img_file.is_file() and img_file.name[-4:] == '.jpg':
        # Step 1: Save Grayscale Image
        img = imread(img_file)
        shutil.copy(img_file, os.path.join(target_dir_img, f'{file_idx}.jpg'))

        # Step 2: Find Bounding Boxes
        class_name_str = '-'.join(img_file.name.split('_')[1:-1])

        try:
            mask_img = imread(os.path.join(ds_images_masks, img_file.name[:-4] + '.png'))
            bounding_boxes, bounding_boxes_str = create_bounding_boxes(mask_img, class_name_str)

            with open(os.path.join(target_dir_text, str(file_idx) + '.txt'), 'w+') as f:
                f.seek(0)
                f.write(bounding_boxes_str)

            plt.imshow(img)
            for box in bounding_boxes:
                xmin, xmax, ymin, ymax = box
                plt.gca().add_patch(
                    Rectangle(xy=(xmin, ymin), width=xmax - xmin, height=ymax - ymin, linewidth=1, edgecolor='r', facecolor='none'))

            plt.savefig(os.path.join(target_dir_boxed, str(file_idx) + '.png'))
            plt.cla()

        except Exception as e:
            print(e)

        # Increment Fish File Index
        file_idx += 1

        print(f"Image Processing {str((file_idx / num_images)*100)[:5]}% Complete")

source_dir = ds_images_empty
for img_file in source_dir.iterdir():
    if img_file.is_file() and img_file.name[-4:] == '.jpg':
        img = Image.open(img_file).convert('L')
        img.save(os.path.join(target_dir_img, f'{file_idx}.jpg'))
        open(os.path.join(target_dir_text, str(file_idx) + '.txt'), 'w+')
        # Increment Fish File Index
        file_idx += 1
print("Finished Processing Grayscale Empty Images")
