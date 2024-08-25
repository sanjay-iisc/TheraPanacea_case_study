import tools.readingData as read
import dataLoader 
import models.vg16 
import models.mobile_v2
import tensorflow as tf

import sys,  argparse, os
import numpy as np
import models.simple
from datetime import datetime
import tools.checkpoint_fun as call_back

from collections import defaultdict
import metrics.custom_metrics as metric
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)




@tf.function
def weighted_binary_cross_entropy_2(y_true, y_pred, weights=[0.87, 0.13]):
    bce_loss=tf.keras.losses.binary_crossentropy(y_true, tf.squeeze(y_pred), from_logits=False)
    weight_vector = y_true * weights[1] + (1.0 - y_true) * weights[0]
    weighted_bce = weight_vector * bce_loss
    loss=tf.reduce_mean(weighted_bce)
    return loss


def weighted_binary_cross_entropy(y_true, y_pred,epsilon_=1e-7, weights=[0.87, 0.13]):
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
    y_true = tf.clip_by_value(y_true, epsilon_, 1.0 - epsilon_)
    weights = tf.convert_to_tensor(weights, dtype=y_pred.dtype)

    A1= weights[0]*(1 - y_true) * tf.math.log(1 - y_pred+ epsilon_)
    A2= weights[1]*y_true*tf.math.log(y_pred+ epsilon_)

    loss=-tf.reduce_mean(A1+A2)
    return loss


def binary_cross_entropy(y_true, y_pred,  epsilon_=1e-7 ):
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
    A1= (1 - y_true) * tf.math.log(1 - y_pred+ epsilon_)
    A2= y_true*tf.math.log(y_pred+ epsilon_)
    loss=-tf.reduce_mean(A1+A2)
    return loss




class TrainClassifier:
    def __init__(self,augment_data=None,lr_rate=None,_image_size=None, _batch_size=None, epochs=None, Dense_units=None,
                  prob_dropout=None, is_base_weights_train=False, model_name="VGG16_Based",_is_checkpoint_loaded=None, data_path="data") -> None:
        # model = models.vg16.VGG16_Based(256, input_shape=(*_image_size, 3), prob_dropout=0.1 ,is_vgg_weights_requ=False)
        # model =models.simple.simple_model()
        self.model_name=model_name
        self.lr_rate= lr_rate
        self.augment_data=augment_data
        # self.accu = accuracy
        self.epochs=epochs
        self.BCe=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self._image_size=_image_size
        self.Dense_units=Dense_units
        self.is_base_weights_train=is_base_weights_train
        self.prob_dropout=prob_dropout
        self._is_checkpoint_loaded=_is_checkpoint_loaded
        self.train_f1score_metric = metric.F1Score()
        self.val_f1score_metric = metric.F1Score()

        self.train_auc_metric=tf.keras.metrics.AUC(curve='ROC')
        self.train_pr_metric=tf.keras.metrics.AUC(curve='PR')
        
        self.val_auc_metric=tf.keras.metrics.AUC(curve='ROC')
        self.val_pr_metric=tf.keras.metrics.AUC(curve='PR')

        train_images_path=read.getTrainDataPathList( data_path+"/train_img")
        labels=read.getTrainLabels(data_path+"/label_train.txt")

        Datasets_init=dataLoader.ImageDataLoader(train_images_path, labels, batch_size= _batch_size,image_size=self._image_size[:-1], augument=self.augment_data)
        self.train_dataset=Datasets_init.get_train_dataset()
        self. val_dataset = Datasets_init.get_val_dataset()
    
    @tf.function
    def apply_gradient(self, model, optimizer ,x, y):
        with tf.GradientTape() as tape:
            logits = model(x)
            # loss_value = binary_cross_entropy(y, logits)
            loss_value = weighted_binary_cross_entropy_2(y, logits)
            # loss_value=self.BCe(y, logits, sample_weight=[0.15])
        
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return logits, loss_value
    
    def one_epoch_val(self, model):
        total=len(self.val_dataset)
        bar_length = int(30)
        losses = []
        for step, (x_batch_val, y_true_val) in enumerate(self.val_dataset):
           logits = model(x_batch_val)
        #    loss_value =self.BCe(y_true_val, y_pred=logits, sample_weight=[1.0])
           loss_value=weighted_binary_cross_entropy_2(y_true_val, logits)
           
           y_pred_val=tf.squeeze(tf.where(logits >= 0.5, 1.0, 0.0))
           
           self.val_f1score_metric.update_state(tf.cast(y_true_val, dtype=tf.int32), tf.cast(y_pred_val, dtype=tf.int32))
           self.val_auc_metric.update_state(tf.cast(y_true_val, dtype=tf.int32), logits)
           self.val_pr_metric.update_state(tf.cast(y_true_val, dtype=tf.int32), logits)


           losses.append(loss_value)
           
           
           percent = 100.0 * step / total
           num_equals = int(percent / (100.0 / bar_length))
           progress_bar = '[' + '=' * num_equals + ' ' * (bar_length - num_equals) + ']'
           sys.stdout.write('\rOne Validation Batch: {} {:>3}% loss:{:.4f}'.format(progress_bar, int(percent), loss_value))
           sys.stdout.flush()
        sys.stdout.write('\n')
        return losses

    
    def one_epoch_train(self, model,optimizer):
        total=len(self.train_dataset)
        bar_length = int(30)
        losses = []
        for step, (x_batch_train, y_true_train) in enumerate(self.train_dataset):
           logits, loss_value = self.apply_gradient(model,optimizer,x_batch_train, y_true_train)
           current_lr = float(optimizer.lr)
           y_pred_train=tf.squeeze(tf.where(logits >= 0.5, 1.0, 0.0))
           
           self.train_f1score_metric.update_state(tf.cast(y_true_train, dtype=tf.int32), tf.cast(y_pred_train, dtype=tf.int32))
           
           self.train_auc_metric.update_state(tf.cast(y_true_train, dtype=tf.int32), logits)
           self.train_pr_metric.update_state(tf.cast(y_true_train, dtype=tf.int32), logits)


           losses.append(loss_value)
           percent = 100.0 * step / total
           num_equals = int(percent / (100.0 / bar_length))
           progress_bar = '[' + '=' * num_equals + ' ' * (bar_length - num_equals) + ']'
           sys.stdout.write('\rOne Training Batch: {} {:>3}% loss:{:.4f} current_lr:{:.6f}'.format(progress_bar, int(percent), loss_value, current_lr))
           sys.stdout.flush() 
        sys.stdout.write('\n')
        return losses
    
   
    def __call__(self):
        start_epoch=0
        end_epoch=self.epochs

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Checkpoint_filename = f"checkpoint/checkpoint_{timestamp}"
        if self.model_name=="VGG16_Based": 
            model = models.vg16.VGG16_Based(self.Dense_units,input_shape=self._image_size, 
                                            prob_dropout=self.prob_dropout,is_vgg_weights_requ=self.is_base_weights_train)
            assert self._image_size==(224,224,3),"It's MobileV2_based make dim as (224,224,3)" 
            if self._is_checkpoint_loaded:
                start_epoch=70
                checkpoint_path="checkpoint/checkpoint_20240825-134313/epoch_70.ckpt"
                model.load_weights(checkpoint_path)
                tf.print("******************VGG16_Based: Checkpoint loaded**********************************")
            else:
                tf.print("******************VGG16_Based**********************************")
        
        elif self.model_name=="MobileV2_Based":
            model =models.mobile_v2.MobileV2_Based(self.Dense_units,input_shape=self._image_size,is_mobile_weights_requ=self.is_base_weights_train)
            tf.print("**************************MobileV2_Based******************")
            # assert self._image_size==(224,224,3),"It's MobileV2_based make dim as (224,224,3) " 
        initial_learning_rate=self.lr_rate
        decay_freauency=2
        decay_steps=len(self.train_dataset)*decay_freauency
        decay_rate=0.8
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                     decay_steps=decay_steps,decay_rate=decay_rate,staircase=True)
        
        optimizer =tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, weight_decay=1e-4)
        checkpoint_callback = call_back.CheckpointCallback(model, checkpoint_path=Checkpoint_filename, save_freq=2)

        for epoch in np.arange(start_epoch, end_epoch, 1):
            tf.print("\nepoch {}/{}".format(epoch+1,self.epochs))
            train_losses = self.one_epoch_train(model, optimizer)
            val_losses =self.one_epoch_val(model)

            checkpoint_callback.on_epoch_end(epoch)
            
            train_F1=self.train_f1score_metric.result()
            val_F1=self.val_f1score_metric.result()

            t_auc=self.train_auc_metric.result()
            t_pr=self.train_pr_metric.result()
            
            v_auc=self.val_auc_metric.result()
            v_pr=self.val_pr_metric.result()
            

            log_string="epoch:{}, train_loss:{:<1.4f}, val_loss:{:<1.4f}, train_F1:{:<1.4f}, val_F1:{:<1.4f}, train_prec:{:<1.4f}, val_prec:{:<1.4f}, train_recall:{:<1.4f}, val_recall:{:<1.4f}, train_auc:{:<1.4f}, val_auc:{:<1.4f}, train_pr:{:<1.4f}, val_pr:{:<1.4f}, train_fpr:{:<1.4f}, val_fpr:{:<1.4f}".format(
                epoch,
                np.mean(train_losses), 
                np.mean(val_losses), 
                train_F1[0].numpy(), 
                val_F1[0].numpy(),
                train_F1[1].numpy(), 
                val_F1[1].numpy(),
                train_F1[2].numpy(), 
                val_F1[2].numpy(),
                t_auc.numpy(), v_auc.numpy(), t_pr.numpy(), v_pr.numpy(), train_F1[3].numpy(), 
                val_F1[3].numpy())
            
            print(log_string)

            with open(f"logs/log_file_{timestamp}_.txt", "+a") as f:
                f.write(log_string+"\n")
            

            self.train_f1score_metric.reset_states()
            self.val_f1score_metric.reset_states()

            self.train_auc_metric.reset_states()
            self.train_pr_metric.reset_states()
            
            self.val_auc_metric.reset_states()
            self.val_pr_metric.reset_states()
            

if __name__=="__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('--LR', type=float, default=0.001, help='learning rate')
        parser.add_argument('--B', type=int, default=8)
        parser.add_argument('--E', type=int, default=2)
        parser.add_argument('--dense_units', type=int, default=16, help='Number of dense units in the vgg_based model')
        parser.add_argument('--image_size', type=int, nargs=3, default=[224, 224, 3], help='(width, height, channels)')
        parser.add_argument('--base_model', type=str, default="VGG16_Based", help='Base model name (VGG16_Based,MobileV2_Based)')
        parser.add_argument('--base_train', type=bool, default=False, help='Base model training is_rquired')
        parser.add_argument('--ProbD', type=float, default=0.5, help='Probability Dropout for Vgg16')
        parser.add_argument('--is_chkpt_load', type=bool, default=False, help='checkpoint loaded')
        parser.add_argument('--is_aug_data', type=bool, default=True, help='checkpoint loaded')
        return parser.parse_args()
    
    args = parse_arguments()

    tf.print(f" Augument data={args.is_aug_data}, "
      f"Is checkpoint loaded={args.is_chkpt_load}, "
      f"Learning rate={args.LR}, "
      f"image_size={tuple(args.image_size)}, "
      f"_batch_size={args.B}, "
      f"epochs={args.E}, "
      f"Dense_units={args.dense_units}, "
      f"prob_dropout={args.ProbD}, "
      f"is_base_weights_train={args.base_train}, "
      f"model_name={args.base_model}")

    
    Train_1=TrainClassifier(augment_data=args.is_aug_data,lr_rate=args.LR, _image_size=tuple(args.image_size), _batch_size=args.B,  epochs=args.E, Dense_units=args.dense_units,
                            prob_dropout=args.ProbD, is_base_weights_train=args.base_train, model_name=args.base_model,_is_checkpoint_loaded=args.is_chkpt_load)()

   
