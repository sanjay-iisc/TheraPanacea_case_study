import tools.readingData as read
import dataLoader 
import models.vg16 
import tensorflow as tf
import random

import metrics.custom_metrics as metric
from sklearn.metrics import confusion_matrix
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
    plt.savefig("confusion_matrix.png")
    plt.show()

def image_processing(image_path,image_size=(224,224)):
    image =tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image= tf.image.resize(image, image_size)
    image= tf.cast(image, tf.float32)/255.0 ### Making it O and 1
    return tf.expand_dims(image, axis=0)

def vgg_based_model():
    model_vgg = models.vg16.VGG16_Based(16,input_shape=(224,224,3))
    checkpoint_path="checkpoint/checkpoint_20240824-212043/epoch_50.ckpt"
    model_vgg.load_weights(checkpoint_path)
    return model_vgg

if __name__=="__main__":
 
    org_image_shape=(64,64)

    inf_model=vgg_based_model()
    test_path = "data/val_img"

    test_img_lists=read.getTestDataPathList(test_path)#random.sample(read.getTestDataPathList(test_path),10)
    print(test_img_lists)
    
    for idx,img_path in enumerate(test_img_lists):
        _img=image_processing(img_path)
        print(idx)
        y_pred=inf_model(_img)
        y_pred_binary=tf.squeeze(tf.where(y_pred >= 0.5, 1.0, 0.0)).numpy()
        y_pred_prob=tf.squeeze(y_pred).numpy()
        with open("test_pred.txt", "+a") as f:
            f.write(f"{y_pred_binary}\n")
    #     plt.subplot(5, 2, idx + 1)
    #     plt.imshow(tf.squeeze(_img))
    
    #     plt.title(f'Predicted: {y_pred_binary:.2f}, {y_pred_prob:.2f}', fontsize=7)
    #     plt.axis('off')
    #     plt.savefig("Prediction_image.png")
    # plt.show()
    #################Testing phase
    
    
    
    
    # #### training whole model
    # train_images_path=read.getTrainDataPathList( "data/train_img")
    # labels=read.getTrainLabels("data/label_train.txt")
    # Datasets_init=dataLoader.ImageDataLoader(train_images_path, labels, batch_size=32,image_size=(224,224))
    # val_dataset =  Datasets_init.get_val_dataset()
    
    # ####
    
    # for index, (x_train, y_true) in enumerate(val_dataset):
    #     y_pred =(model_vgg((x_train)))
    #     # print(y_pred)
    #     y_pred=tf.squeeze(tf.where(y_pred >= 0.5, 1.0, 0.0))
    #     # print(y_pred)
    #     # print(y_true)
    #     plot_confusion_matrix(tf.cast(y_true, dtype=tf.int32), tf.cast(y_pred, dtype=tf.int32))
    #     break;
    # test_F1Score = metric.F1Score()
    
    # test_F1Score.update_state(tf.cast(y_true, dtype=tf.int32), tf.cast(y_pred, dtype=tf.int32))
    # print(test_F1Score.result())
    # test_F1Score.reset_states()
    # ###
    
    # # for index, (x_train, y_true) in enumerate(val_dataset):
    # #     y_pred =(model_vgg((x_train)))
    

    # #     AUC_train=tf.keras.metrics.AUC(curve='ROC')
    # #     PR_train=tf.keras.metrics.AUC(curve='PR')
        
    # #     AUC_train.update_state(tf.cast(y_true, dtype=tf.int32), y_pred)
    # #     PR_train.update_state(tf.cast(y_true, dtype=tf.int32), y_pred)
    # #     print(AUC_train.result())
    # #     print(PR_train.result())

    # #     AUC_train.reset_state()
    # #     PR_train.reset_state()

