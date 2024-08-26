import tools.readingData as read
import dataLoader 
import models.vg16 
import tensorflow as tf
import random
import numpy as np

import metrics.custom_metrics as metric
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,det_curve, precision_recall_curve
import matplotlib.pyplot as plt
import tools.readingData as read

def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = 'd'
    thresh = cm.max() / 2.
    plt.savefig("pictures/confusion_matrix.png")
    plt.show()

def image_processing(image_path,image_size=(224,224)):
    image =tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image= tf.image.resize(image, image_size)
    image= tf.cast(image, tf.float32)/255.0 ### Making it O and 1
    return tf.expand_dims(image, axis=0)

def vgg_based_model(checkpoint_path):
    model_vgg = models.vg16.VGG16_Based(16,input_shape=(224,224,3))
    
    model_vgg.load_weights(checkpoint_path)
    return model_vgg


def create_pred_text(image_path_list, inf_model):
    for idx,img_path in enumerate(image_path_list):
        _img=image_processing(img_path)
        print(idx)
        y_pred=inf_model(_img)
        y_pred_binary=tf.squeeze(tf.where(y_pred >= 0.82, 1.0, 0.0)).numpy()
        y_pred_prob=tf.squeeze(y_pred).numpy()
        with open("test_pred.txt", "+a") as f:
            f.write(f"{y_pred_binary}\n")

def predicated_img(image_path_list, inf_model):
    for idx,img_path in enumerate(image_path_list):
        _img=image_processing(img_path)
        print(idx)
        y_pred=inf_model(_img)
        y_pred_binary=tf.squeeze(tf.where(y_pred >= 0.5, 1.0, 0.0)).numpy()
        y_pred_prob=tf.squeeze(y_pred).numpy()
        plt.subplot(5, 2, idx + 1)
        plt.imshow(tf.squeeze(_img))
    
        plt.title(f'Predicted: {y_pred_binary:.2f}, {y_pred_prob:.2f}', fontsize=7)
        plt.axis('off')
        plt.savefig("pictures/Prediction_image.png")

def get_yp_yt(val_dataset, inf_model):
    pred_list=[]
    true_list=[]
    for index, (x_train, y_true) in enumerate(val_dataset):
        y_pred =(inf_model((x_train)).numpy())
        pred_list.append(y_pred.flatten())
        true_list.append(tf.cast(y_true, dtype=tf.int32).numpy().flatten())
        print(index)
        if index>2500:
            break
    return np.array(pred_list).flatten(), np.array(true_list).flatten()

def plot_roc(Tr, Pr):
    fpr, tpr, thresholds = roc_curve(Tr, Pr, pos_label=1)
    ras=roc_auc_score(Tr, Pr)
    # print(thresholds)
    # print("fpr", fpr)
    # print("tpr", tpr)
    max_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[max_idx]

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC-AUC: {ras:.2f}')
    plt.plot(fpr[max_idx], tpr[max_idx], 'ro', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.grid()
    plt.legend()
    plt.savefig("pictures/ROC_curve.png")

def plot_det(Tr, Pr):
    fpr, fnr, thresholds = det_curve(Tr, Pr, pos_label=1)
    max_idx = np.argmin(abs(fnr - fpr))
    optimal_threshold = thresholds[max_idx]

    plt.figure()
    plt.plot(fpr, fnr, label=f'Detection curve')
    plt.plot(fpr[max_idx], fnr[max_idx], 'ro', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.title("Detection curve")
    plt.grid()
    plt.legend()
    plt.savefig("pictures/DET_curve.png")

    HTER=(fnr+fpr)/2
    idx = np.argmin(HTER)
    plt.figure()
    plt.plot(thresholds, (fnr+fpr)/2, label=f'Detection curve')
    plt.plot(thresholds[idx], HTER[idx], 'ro', label=f'Optimal Threshold: {thresholds[idx]:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("HTER")
    plt.title("Detection curve")
    plt.grid()
    plt.legend()
    plt.savefig("pictures/HTER_curve.png")
    
def plot_PR(Tr, Pr):
    precision, recall, thresholds = precision_recall_curve(Tr, Pr, pos_label=1)
  
    max_idx = np.argmin(abs(recall-precision))
    optimal_threshold = thresholds[max_idx]

    plt.figure()
    plt.plot(precision, recall, label=f'P-R')
    plt.plot(precision[max_idx], recall[max_idx], 'ro', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("precision-recall Curve")
    plt.grid()
    plt.legend()
    plt.savefig("pictures/PR-Recall_curve.png")
    
    F1= 2*((precision*recall)/(precision+recall))
    index_max=np.argmax(F1)
    
    plt.figure()
    plt.plot(thresholds, F1[:-1], label=f'P-R')
    plt.plot(thresholds[index_max], F1[index_max], 'ro', label=f'Optimal Threshold: {thresholds[index_max]:.2f}')
    plt.xlabel("Thresholds")
    plt.ylabel("F1")
    plt.title("F1 score Curve")
    plt.grid()
    plt.legend()
    plt.savefig("pictures/F1_Threshold_curve.png")


if __name__=="__main__":
 
    org_image_shape=(64,64)
    checkpoint_path="checkpoint/checkpoint_20240825-172224/epoch_100.ckpt"
    inf_model=vgg_based_model(checkpoint_path)
    test_path = "data/val_img"

    test_img_lists=random.sample(read.getTestDataPathList(test_path),10)#read.getTestDataPathList(test_path)#random.sample(read.getTestDataPathList(test_path),10)
    
    # predicated_img(test_img_lists, inf_model)
    
    #################Testing phase
    
    
    
    
    #### training whole model
    train_images_path=read.getTrainDataPathList( "data/train_img")
    labels=read.getTrainLabels("data/label_train.txt")
    Datasets_init=dataLoader.ImageDataLoader(train_images_path, labels, batch_size=8,image_size=(224,224))
    val_dataset =  Datasets_init.get_val_dataset()

    
    Pr, Tr=get_yp_yt(val_dataset, inf_model)
    plot_roc(Tr, Pr) 
    plot_PR(Tr, Pr)
    plot_det(Tr, Pr)
    cm = confusion_matrix(Tr, np.where(Pr >= 0.52, 1, 0))
    print(cm)


