# "HAM: Hidden Anchor Mechanism for Scene Text Detection"
The code of ["HAM: Hidden Anchor Mechanism for Scene Text Detection"](https://ieeexplore.ieee.org/document/9143459).

We follow the code of [EAST](https://github.com/argman/EAST).

## Requirements:  
tensorflow >=1.13  
python3  
Polygon  
sklearn  
Shapely  
Pillow  
numpy  
scipy
cython

## Checkpoints
Checkpoint for ICDAR 2015 is avaiable at [ICDAR 2015](https://drive.google.com/file/d/1GqwBMcvQ28QeWeuRnlg_w6F5iACHoAFo/view?usp=sharing), the checkpoint is pre-trained on ICDAR MLT 2017, ICDAR 2015, ICDAR 2013, and finetuned on ICDAR 2015. Unzip it and move the three files to the ./checkpoints folder.


## Make
before runing the code, please:  
cd lanms; make; cd ../   

## Run  

### without_IRB: sh run.sh [gpu_id]
### withIRB: sh run_IRB.sh [gpu_id]
