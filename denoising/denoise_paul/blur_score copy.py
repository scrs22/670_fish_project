import os
import utils

ds = '.DS_Store'
dir_name = 'dataset_cropped_split_blurScore'
threshold = 0.002946554451218414
blur_count = {'train': 0, 'test':0, 'val':0}

for split in os.listdir(os.path.join(dir_name)):
    if split == ds:
        continue
    root = os.path.join(dir_name, split)
    for fish_type in os.listdir(root):
        if fish_type == ds:
            continue
        fish_dir = os.path.join(root, fish_type, 'images')
        blur_dir = os.path.join(root, fish_type, 'blur')
        if not os.path.exists(blur_dir):
            os.mkdir(blur_dir)
        # get names of each image file
        fishImg_names = [f for f in os.listdir(fish_dir) if os.path.isfile(os.path.join(fish_dir, f))]
        print(f'Currently at {fish_type}')
        
        # iterate through all fish image
        for img_name in fishImg_names:
            img = utils.imread(os.path.join(fish_dir, img_name))
            score = utils.blurScore(img)
            if score < threshold:
                prediction = 'blur'
                blur_count[split] = blur_count[split]+1
            else:
                prediction = 'clear'
            # save the score and prediction in txt file
            blur_name = img_name.split('.')[0] + '.txt'
            # TODO uncommon below code to write txt files
            # with open(os.path.join(blur_dir, blur_name), 'w') as f:
            #     f.write(f'{score} {prediction}')

print(f'Done: blur count {blur_count}')