import os,sys
# sys.path.append(os.path.abspath("."))
import warnings
import tensorflow as tf


class MobileV2_Based(tf.keras.Model):
    def __init__(self , units, input_shape=(224, 224,3) ,is_mobile_weights_requ=False):
        
        super( MobileV2_Based, self ).__init__(name  ='MobileV2_Based')  
        self.mobilev2_base=tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        self.mobilev2_base.trainable=is_mobile_weights_requ

        self.GAP = tf.keras.layers.GlobalAveragePooling2D()

        self.dense = tf.keras.layers.Dense(units, activation='relu')

        self.classifier=tf.keras.layers.Dense(1, activation='sigmoid')
        # assert input_shape==(224,224,3),"It's MobileV2_based make dim as(224,224,3) " 

    def call(self,input):
        x= self.mobilev2_base(input)
        x= self.GAP(x)
        x= self.dense(x)
        x = self.classifier(x)
        return x


if __name__=="__main__":
    input_shape = (224,224,3)
    mobilev2_base=tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    mobilev2_base.summary()
    input=tf.ones((32, *input_shape))
    model_class = MobileV2_Based(1000)
    y_pred = model_class(input)
    print(model_class.summary())
  
