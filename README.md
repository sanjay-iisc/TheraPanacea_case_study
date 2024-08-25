# TheraPanacea Case Study: Binary Classification Problem

## Problem Description
The objective of this case study is to classify images based on whether they include a hat/spec or not, using only the face. We are exploring shallow learning techniques, but will eventually transition to Convolutional Neural Networks (CNNs) given their state-of-the-art performance in image classification tasks.

## System Configuration
- **Operating System**: Linux
- **TensorFlow Version**: 12.5
- **CUDA Version**: 12.2
- **cuDNN Version**: 8.9.7

## TensorFlow Installation with CUDA
To install TensorFlow with CUDA support, avoid using the `tensorflow[and-cuda]` pip option as it may lead to errors. Instead, use the following installation steps:

1. Install TensorFlow (compatible version):
   ```bash
   python3 -m pip install tensorflow==12.5



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