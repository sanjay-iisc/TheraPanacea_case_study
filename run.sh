#!/bin/bash
source ~/TheraEnv/bin/activate
# python3 train.py --LR 0.001 --image_size 224 224 3 --B 8 --E 50 --dense_units 16 #checkpoint_20240824-212043
# python3 tools/reading_plotting_Logfile.py
## second run on 25/7am
# python3 train.py --LR 0.001 --B 16 --E 200 --dense_units 1000 --image_size 64 64 3 --base_model MobileV2_Based #checkpoint_20240825-073243
### third run--checkpoint_20240825-134313
# python3 train.py --LR 0.001 --image_size 224 224 3 --B 16 --E 100 --dense_units 16 --base_model "VGG16_Based" --is_chkpt_load True 
### 4 
# python3 train.py --LR 0.001 --image_size 224 224 3 --B 16 --E 100 --dense_units 16 --base_model "VGG16_Based" --is_chkpt_load True --is_aug_data True

## run 5
# python3 train.py --LR 0.001 --imagesize 224 224 3 --B 16 --E 80 --denseunits 16 --basemodel "VGG16_Based" --ischkptload false --isaugdata false

# python train.py --LR 0.001 --B 8 --E 80 --denseunits 16 --imagesize 224 224 3 --basemodel "VGG16_Based" --basetrain False --ProbD 0.5 --ischkptload False --isaugdata True
# python train.py --LR 0.00065 --B 8 --E 80 --denseunits 16 --imagesize 224 224 3 --basemodel "VGG16_Based" checkpoint_20240826-175517

 python train.py --LR 0.00065 --B 8 --E 80 --denseunits 16 --imagesize 224 224 3 --basemodel "VGG16_Based"
