
import numpy as np
import logging
import tools.readingData as read

from sklearn.model_selection import train_test_split
import tensorflow as tf

class ImageDataLoader:
    def __init__(self, image_paths, labels, batch_size=32, image_size=(224,224), shuffle=True, validation_split=0.2):
        """image_paths, labelPaths are in lists"""
        self.image_paths=image_paths
        self.labels=labels
        self.batch_size=batch_size
        self.image_size=image_size
        self.is_shuffle=shuffle
        self.validation_split=validation_split

        self.train_paths, self.val_paths, self.train_labels, self.val_labels = self._split_paths_data()

        self.train_dataset = self._build_dataset(self.train_paths, self.train_labels)
        self.val_dataset = self._build_dataset(self.val_paths, self.val_labels)
    
    
    def _load_image(self, imagePath, label):
        image =tf.io.read_file(imagePath)

        image = tf.image.decode_jpeg(image, channels=3)
        # image= tf.image.resize(image, self.image_size)
        image= tf.cast(image, tf.float32)/255.0 ### Making it O and 1
        return image, label
    
    def _split_paths_data(self):
        train_paths, val_paths, train_labels, val_labels = train_test_split(self.image_paths, self.labels, 
                                                                            test_size=self.validation_split, stratify=self.labels, random_state=42)
        return train_paths, val_paths, train_labels, val_labels 
    
    def _build_dataset(self, img_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        dataset = dataset.map(self._load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.is_shuffle:
            dataset = dataset.shuffle(buffer_size=len(img_paths))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset
    
    def __len__(self):
        return len(self.image_paths)
    
    def __repr__(self):
        return f"# of images ={len(self.image_paths)}"



if __name__=="__main__":
    train_images_path=read.getTrainDataPathList( "data/train_img")
    labels=read.getTrainLabels("data/label_train.txt")
    DataLoader=ImageDataLoader(train_images_path, labels)
    train_dataset= DataLoader.get_train_dataset()
    val_dataset =DataLoader.get_val_dataset()

    print("length of train datasets = ",len(train_dataset)*32)
    print("length of val datasets = ", len(val_dataset)*32)
    for index,(x_train, y_train) in enumerate(train_dataset):
        if index==0:
            print(x_train.shape)
        assert x_train.shape==(32,64,64,3)
    
    for index,(x_val, y_val) in enumerate(val_dataset):
        if index==0:
            print(x_val.shape)
        assert x_val.shape==(32,64,64,3)



