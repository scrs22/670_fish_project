from pylabel import importer

folders=['train','test','val']
for folder in folders:
    path_to_annotations = f"dataset_single_folder_train_val_test_split/{folder}/labels"

    #Identify the path to get from the annotations to the images 
    path_to_images = "../images"

    #Import the dataset into the pylable schema 
    #Class names are defined here https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
    classes = (['scallop','herring','dead-scallop','flounder','roundfish','skate'])
    dataset = importer.ImportYoloV5(path_to_annotations,"png",classes,path_to_images,f"{folder}")

    dataset.export.ExportToCoco(cat_id_index=1)