import os,sys
# sys.path.append(os.path.abspath("."))
import warnings
import tensorflow as tf


class VGG16_Based(tf.keras.Model):
    def __init__(self , units, input_shape=(224,224,3), prob_dropout=0.5 ,is_vgg_weights_requ=False):
        super( VGG16_Based, self ).__init__(name  ='VGG16_Based')  
        self.vgg_base=tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        self.vgg_base.trainable=is_vgg_weights_requ
        self.dense = tf.keras.layers.Dense(units, activation='relu')
        self.flat=tf.keras.layers.Flatten()
        self.drop=tf.keras.layers.Dropout(prob_dropout)
        self.classifier=tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self,input):
        x= self.vgg_base(input)
        x= self.flat(x)
        x= self.dense(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x


if __name__=="__main__":
    input_shape = (224,224,3)
    vgg_base=tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg_base.summary()
    input=tf.ones((32, *input_shape))
    model_class = VGG16_Based(256)
    y_pred = model_class(input)
    print(model_class.summary())
  

