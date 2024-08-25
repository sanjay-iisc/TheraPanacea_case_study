#!/bin/bash
source ~/env/bin/activate
# python3 train.py --LR 0.001 --image_size 224 224 3 --B 8 --E 50 --dense_units 16 #checkpoint_20240824-212043
# python3 tools/reading_plotting_Logfile.py
## second run on 25/7am
# python3 train.py --LR 0.001 --B 16 --E 200 --dense_units 1000 --image_size 64 64 3 --base_model MobileV2_Based #checkpoint_20240825-073243
### third run
# python3 train.py --LR 0.001 --image_size 224 224 3 --B 8 --E 100 --dense_units 16 --base_model "VGG16_Based" --is_chkpt_load True #log_file_20240825-104217_
##fourth run
# checkpoint_20240825-145759
# python3 train.py --LR 0.001 --B 16 --E 200 --dense_units 1000 --image_size 64 64 3 --base_model MobileV2_Based

#5---log_file_20240825-165455_
# python3 train.py --LR 0.001 --B 16 --E 200 --dense_units 256 --image_size 64 64 3 --base_model MobileV2_Based --is_aug_data True

python3 test.py