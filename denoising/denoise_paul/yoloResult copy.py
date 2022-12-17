import os
import utils

ds = '.DS_Store'
test_name = 'dataset_cropped_split_blurScore/test'
predicted_name = 'dataset_cropped_split_blurScore/predicted/detect'
type_label = ['scallop','herring','dead-scallop','flounder','roundfish','skate']
result = {'TBlur': 0, 'FBlur':0, 'TClear':0, 'FClear': 0}
FalseClearScore = []

for type_name in os.listdir(os.path.join(predicted_name)):
    if type_name == ds:
        continue
    labelsPath = os.path.join(predicted_name, type_name, 'labels')
    for predicted in os.listdir(labelsPath):
        if predicted == ds:
            continue
        file1 = open(os.path.join(labelsPath, predicted), 'r')
        line = file1.readline()
        predicted_label = type_label[int(line.split(' ')[0])]
        
        # Get blur result
        blur_result = os.path.join(test_name, type_name, 'blur', predicted)
        file1 = open(blur_result, 'r')
        line = file1.readline().split(' ')
        blur_score, blur_label = line[0], line[1]
        
        if type_name == predicted_label:
            # True Blur
            if blur_label == 'blur':
                result['TBlur'] += 1
            else:
                result['TClear'] += 1
        else:
            if blur_label == 'blur':
                result['FBlur'] += 1
            else:
                FalseClearScore.append(blur_score)
                result['FClear'] += 1

print(result)
print(sorted(FalseClearScore))