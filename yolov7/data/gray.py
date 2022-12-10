import cv2

import os

from os import listdir,makedirs

from os.path import isfile,join

folders=['test','train','val']
species=['acanthopagrus',  'epinephelus',  'f2'  ,'f3',  'f6' , 'gerres' , 'lutjanus']

for folder in folders:
    for specie in species:

          path = f'/home/ubuntu/herring_670/670_fish_project/yolov7/data/input/{folder}/{specie}/images' # Source Folder
          dstpath = f'/home/ubuntu/herring_670/670_fish_project/yolov7/data/input/{folder}/{specie}/images' # Destination Folder
          try:
              makedirs(dstpath)
          except:
              print ("Directory already exist, images will be written in same folder")
          # Folder won't used
          files = list(filter(lambda f: isfile(join(path,f)), listdir(path)))
          for image in files:
              try:
                  img = cv2.imread(os.path.join(path,image))
                  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                  dstPath = join(dstpath,image)
                  cv2.imwrite(dstPath,gray)
              except:
                  print ("{} is not converted".format(image))
              