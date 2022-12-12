import os
import shutil
from pathlib import Path

uncropped_split_dataset_root = Path('./input')
target_unified_root = Path('./dataset_single_folder')
target_unified_train_val_test_root = Path('./dataset_single_folder_train_val_test_split')

processed = 0
for fish_file in uncropped_split_dataset_root.glob('**/*'):
    if fish_file.is_file() and fish_file.name[-4:] == '.png':
        target_folder = target_unified_root / Path('images')
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy(fish_file, target_folder)
        processed += 1
    if fish_file.is_file() and fish_file.name[-4:] == '.txt':
        target_folder = target_unified_root / Path('labels')
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy(fish_file, target_folder)
        processed += 1

    if processed % 100 == 0:
        print(f"Processed {processed} Files in Unified Dataset")

processed = 0
for set_name in ['train', 'val', 'test']:
    for fish_file in Path(uncropped_split_dataset_root / set_name).glob('**/*'):
        if fish_file.is_file() and fish_file.name[-4:] == '.png':
            target_folder = Path(target_unified_train_val_test_root / set_name / 'images')
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy(fish_file, target_folder)
            processed += 1
        if fish_file.is_file() and fish_file.name[-4:] == '.txt':
            target_folder = Path(target_unified_train_val_test_root / set_name / 'labels')
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy(fish_file, target_folder)
            processed += 1

        if processed % 100 == 0:
            print(f"Processed {processed} Files in Train/Val/Test Dataset")
