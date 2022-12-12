from pylabel import importer


path_to_annotations = "/home/ubuntu/670_fish_project/yolov7/data/dataset_single_folder/labels"

#Identify the path to get from the annotations to the images 
path_to_images = "/home/ubuntu/670_fish_project/yolov7/data/dataset_single_folder/images"

#Import the dataset into the pylable schema 
#Class names are defined here https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
yoloclasses = ['scallop','herring','dead-scallop','flounder','roundfish','skate' ]
dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses,
    img_ext="png", name="fish")

dataset.export.ExportToCoco(cat_id_index=1)