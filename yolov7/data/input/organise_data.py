# import required module
from PIL import Image
  
classes=['skate','scallop','dead scallop','carp','herring','flounder','roundfish']
folders=['train','val','test']
filepath = "geeksforgeeks.png"
img = Image.open(filepath)
  
# get width and height
width,height = img.size