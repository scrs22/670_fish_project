from pathlib import Path
import glob, os
from PIL import Image
dic_fish={'scallop':'0','herring':'1','dead-scallop':'2','flounder':'3','roundfish':'4','skate':'5','acanthopagrus':'6' ,'epinephelus':'7' , 'f2':'8' , 'f3':'9' , 'f6':'10', 'gerres':'11' , 'lutjanus':'12'}
list_dirs=['train','test','val']

for fish in dic_fish:
    for dir_i in list_dirs:
        d_type=dir_i
        fish_type=fish
        fish_idx=dic_fish[fish]
        direc=f'/Users/schanumolu/Desktop/670/project/670_fish_project/yolov7/data/input/{d_type}/{fish_type}/labels'
        im_direc=f'/Users/schanumolu/Desktop/670/project/670_fish_project/yolov7/data/input/{d_type}/{fish_type}/images'

        os.chdir(direc)
        for file_path in glob.glob("*.txt"):
            file_name=file_path.split('.')[0]
            img=Image.open(f'{im_direc}/{file_name}.png')
            w,h=img.shape[0],img.shape[1]
            with open(f'{direc}/{file_path}') as fp:
                lines = fp.readlines()
            norm_lines=[]
            for line in lines:
            
                l=line.split()
                l[1],l[3]=str(float(l[1])/w),str(float(l[3])/w)
                l[2],l[4]=str(float(l[2])/h),str(float(l[4])/h)
                txt=' '.join(l)
                norm_lines.append(txt)
            with open(f'{direc}/{file_path}', 'w') as f:
              for line in norm_lines:
                  f.write(line)
                  f.write('\n')