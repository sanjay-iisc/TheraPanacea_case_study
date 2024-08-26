# TheraPanacea Case Study: Binary Classification Problem

## Problem Description
The objective of this case study is to develop a binary classifier that determines whether images contain specific accessories (such as hats, glasses, etc.) based solely on facial features. The goal is to minimize the Half-Total Error Rate (HTER).

### Dataset Overview
- **Training Data**: Contains 100,000 labeled images.
- **Validation Data**: A separate validation set of 20,000 unlabeled images is provided in the `val_imag` folder.
- **Train/Validation Split**: The training dataset is split into 80% for training and 20% for validation.

![Dataset Overview](pictures/data_image.png)

### Data Distribution
<div align="center">
	<img src="pictures/data_dist.png" alt="Data Distribution">
</div>

The dataset is highly imbalanced, with fewer instances of the "0" label compared to the "1" label in both the training and validation sets. To address this, a weighted approach has been applied to train the classifier.

## System Configuration
- **Operating System**: Linux
- **TensorFlow Version**: 2.15
- **CUDA Version**: 12.2
- **cuDNN Version**: 8.9.7

## Training

To set up the environment and start training, run the following commands:

```bash
cd ~
python3 -m venv env
source ~/env/bin/activate
python3 train.py --LR 0.001 --B 16 --E 200 --dense_units 256 --image_size 224 224 3 --base_model VGG16_Based --is_aug_data True
```
`train.py` is written in lower api of tensorflow and variable is self explainatory and weighted cross entropy function is used for loss to try to address the imblanced data. The base model is VGG16 trained weight on `imagenet`

### HTER
**False Acceptance Rate (FAR):**
$$\text{FAR} =  \frac{\text{FP}}{\text{TN} + \text{FP}}$$

**False Rejection Rate (FRR):**

$$\text{FRR} = \frac{\text{FN}}{\text{TP} + \text{FN}}$$

$$\text{HTER}=\frac{\text{FRR}+\text{FAR}}{2}$$

### ROC curve:

<div align="center">
	<img src="pictures/ROC_curve.png" alt="ROC">
</div>


### Detection curve:

<div align="center">
	<img src="pictures/DET_curve.png" alt="Detection curve">
</div>


### HTER curve:

<div align="center">
	<img src="pictures/HTER_curve.png" alt="HTER curve">
</div>

As required in cased, the threshold is based on HTER to minimize it.

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