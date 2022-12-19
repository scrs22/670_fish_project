from pathlib import Path

import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split

ds_old = Path('./dataset_gs/split')
ds_split = Path('./dataset_deepfish_split')

train_dataset_split_size = 0.6
val_dataset_split_size = 0.2
test_dataset_split_size = 0.2


def create_split_dataset(unsplit_dataset_root, split_dataset_root):
    def split_fish_class_on_indices(indices, class_name, split_name):
        save_img_target_dir = os.path.join(class_name, 'images/')
        save_data_target_dir = os.path.join(class_name, 'labels/')

        target_split_folder = os.path.join(split_dataset_root, split_name, Path(class_name).name)
        target_split_folder_image_root = os.path.join(target_split_folder, 'images/')
        target_split_folder_data_root = os.path.join(target_split_folder, 'labels/')

        os.makedirs(target_split_folder_image_root, exist_ok=True)
        os.makedirs(target_split_folder_data_root, exist_ok=True)

        img_file_list = sorted(list(Path(save_img_target_dir).iterdir()))
        data_file_list = sorted(list(Path(save_data_target_dir).iterdir()))

        if len(indices) <= 1:
            print("Not Enough Indices!")

        for index in indices:
            try:
                shutil.copy(img_file_list[index], target_split_folder_image_root)
                shutil.copy(data_file_list[index], target_split_folder_data_root)
            except:
                print(f"Unable To Copy {class_name} for {index}.png/txt")
                continue

    for fish_type in Path(unsplit_dataset_root).iterdir():
        if fish_type.is_dir():
            fish_images_path = os.path.join(fish_type, 'images')
            num_fish_images = len([x for x in os.listdir(fish_images_path) if x[-4:] == '.png'])
            fish_indices = np.arange(num_fish_images)
            train_indices, val_test_indices = train_test_split(
                fish_indices,
                train_size=train_dataset_split_size,
                random_state=1337
            )
            val_indices, test_indices = train_test_split(
                val_test_indices,
                train_size=val_dataset_split_size/(val_dataset_split_size+test_dataset_split_size),
                random_state=1338
            )
            split_fish_class_on_indices(train_indices, fish_type, 'train')
            split_fish_class_on_indices(val_indices, fish_type, 'val')
            split_fish_class_on_indices(test_indices, fish_type, 'test')

# Create Split Dataset For Cropped Images
create_split_dataset(ds_old, ds_split)