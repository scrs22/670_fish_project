from pathlib import Path
import glob, os
dic_fish={'scallop':'0','herring':'1','dead-scallop':'2','flounder':'3','roundfish':'4','skate':'5'}
# {'acanthopagrus':'0' ,'epinephelus':'1' , 'f2':'2' , 'f3':'3' , 'f6':'4', 'gerres':'5' , 'lutjanus':'6'}
#'
list_dirs=['train','test','val']

for fish in dic_fish:
    for dir_i in list_dirs:
        d_type=dir_i
        fish_type=fish
        fish_idx=dic_fish[fish]
        direc=f'/home/ubuntu/670_fish_project/yolov7/data/denoise/{d_type}/{fish_type}/labels'

        os.chdir(direc)
        for file_path in glob.glob("*.txt"):
            file = Path(file_path)
            file.write_text(file.read_text().replace('dead-scallop', dic_fish['dead-scallop']))
            file.write_text(file.read_text().replace('scallop', dic_fish['scallop']))
            file.write_text(file.read_text().replace('herring', dic_fish['herring']))
            
            file.write_text(file.read_text().replace('flounder', dic_fish['flounder']))
            file.write_text(file.read_text().replace('roundfish', dic_fish['roundfish']))
            file.write_text(file.read_text().replace('skate', dic_fish['skate']))

        