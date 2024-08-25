#!/bin/bash
source ~/TheraEnv/bin/activate
# python3 train.py --LR 0.001 --image_size 224 224 3 --B 8 --E 50 --dense_units 16 #checkpoint_20240824-212043
# python3 tools/reading_plotting_Logfile.py
## second run on 25/7am
# python3 train.py --LR 0.001 --B 16 --E 50 --dense_units 1000 --image_size 64 64 3 --base_model MobileV2_Based #checkpoint_20240825-073243
### third run
python3 train.py --LR 0.001 --image_size 224 224 3 --B 16 --E 100 --dense_units 16 --base_model "VGG16_Based" --is_chkpt_load True #log_file_20240825-104217_