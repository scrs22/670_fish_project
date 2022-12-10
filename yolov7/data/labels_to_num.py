from pathlib import Path
import glob, os
dic_fish={'scallop':'0','herring':'1','dead-scallop':'2','flounder':'3','roundfish':'4','skate':'5','acanthopagrus':'6' ,'epinephelus':'7' , 'f2':'8' , 'f3':'9' , 'f6':'10', 'gerres':'11' , 'lutjanus':'12'}
list_dirs=['train','test','val']

for fish in dic_fish:
    for dir_i in list_dirs:
        d_type=dir_i
        fish_type=fish
        fish_idx=dic_fish[fish]
        direc=f'/home/ubuntu/670_fish_project/yolov7/data/input/{d_type}/{fish_type}/labels'

        os.chdir(direc)
        for file_path in glob.glob("*.txt"):
            file = Path(file_path)
            file.write_text(file.read_text().replace(fish_type, fish_idx))
            file.write_text(file.read_text().replace(fish_type.capitalize(), fish_idx))

        