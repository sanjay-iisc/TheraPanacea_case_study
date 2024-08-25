# TheraPanacea Case Study: Binary Classification Problem

## Problem Description
The objective of this case study is to classify images based on whether they include a hat/spec/..etc or not, using only the face. 

## System Configuration
- **Operating System**: Linux
- **TensorFlow Version**: 2.15
- **CUDA Version**: 12.2
- **cuDNN Version**: 8.9.7

## TensorFlow Installation with CUDA
Avoid using the `tensorflow[and-cuda]` pip option as it may lead to errors (tensorflow==2.17, new version). Instead, use the following installation steps:




['data/train_img/055135.jpg', 'data/train_img/074877.jpg', 'data/train_img/015414.jpg', 'data/train_img/080523.jpg', 'data/train_img/077587.jpg', 'data/train_img/086010.jpg', 'data/train_img/050029.jpg', 'data/train_img/083638.jpg', 'data/train_img/037938.jpg', 'data/train_img/086395.jpg']


TensorFlow with cuda with pip
python3 -m pip install tensorflow[and-cuda]---> please dont use it, have lots error. It has tensorflow==12.7


checked the device with: python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

source myThera/bin/activate;


![alt text](pictures/data_image.png)

![alt text](pictures/data_dist.png)

Number of zeros: 0.121025
Number of ones: 0.878975

to train: ./run 
to test: python3 test.py