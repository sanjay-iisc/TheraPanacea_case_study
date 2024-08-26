# TheraPanacea Case Study: Binary Classification Problem

## Problem Description
The objective of this case study is to classify images based on whether they include a hat/spec/..etc or not, using only the face. 
![alt text](pictures/data_image.png)

## System Configuration
- **Operating System**: Linux
- **TensorFlow Version**: 2.15
- **CUDA Version**: 12.2
- **cuDNN Version**: 8.9.7

## Data Distribution
<div align="center">
	<img src="pictures/data_dist.png">
</div>




## Training

cd ~
python3 -m venv env

source ~/env/bin/activate;

python3 train.py --LR 0.001 --B 16 --E 200 --dense_units 256 --image_size 64 64 3 --base_model MobileV2_Based --is_aug_data True

## Testing
Jupyter file test_i.ipynb and python script named test.py: can be run by python3 test.py






#


Number of zeros: 0.121025
Number of ones: 0.878975

to train: ./run 
to test: python3 test.py


thre:0.82
[[ 2084   336]
 [ 2130 15450]]

thre: 0.38
 [[ 1385  1035]
 [  403 17177]]

thre: 0.6
[[ 1769   651]
 [  982 16598]]

thresh: 0.84
[[ 2111   309]
 [ 2320 15260]]

threshold:0.7
[[ 1918   502]
 [ 1391 16189]]

 thr:0.52
 [[ 1650   770]
 [  746 16834]]