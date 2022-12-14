import os.path
import shutil
from pathlib import Path

image_root = Path('dataset_gs/all/images')
data_root = Path('dataset_gs/all/labels')

target_root = Path('dataset_gs/split/')

for image_file in image_root.iterdir():
    if image_file.is_file():
        image_file_index = image_file.name[:-4]
        class_name = "empty"

        target_data_file = os.path.join(data_root, image_file_index+'.txt')
        data_file_exists = False
        if Path(target_data_file).is_file():
            with open(target_data_file, 'r+') as f:
                line = f.readline()
                if len(line) == 0 or line.index(' ') == -1:
                    continue
                class_name = line[:line.index(' ')]
            data_file_exists = True

        class_name = class_name.lower()

        target_img_folder = os.path.join(target_root, class_name, 'images')
        target_label_folder = os.path.join(target_root, class_name, 'labels')

        os.makedirs(target_img_folder, exist_ok=True)
        os.makedirs(target_label_folder, exist_ok=True)

        shutil.copy(image_file, Path(os.path.join(target_img_folder, (str(image_file_index) + '.png'))))
        if data_file_exists:
            shutil.copy(target_data_file, Path(os.path.join(target_label_folder, (str(image_file_index) + '.txt'))))
