
import numpy as np
import logging
import tools.readingData as read
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

class ImageDataLoader:
    def __init__(self, image_paths, labels, batch_size=None, image_size=None, shuffle=True, augument=None, validation_split=0.2):
        """image_paths, labels are in the lists"""
        self.image_paths=image_paths
        self.labels=labels
        self.batch_size=batch_size
        self.image_size=image_size
        self.is_shuffle=shuffle
        self.validation_split=validation_split
        self.augument=augument

        self.train_paths, self.val_paths, self.train_labels, self.val_labels = self._split_paths_data()

        print(self.train_paths[:10])
        # self.train_dataset = self._build_dataset(self.train_paths[:2500], self.train_labels[:2500])
        # self.val_dataset = self._build_dataset(self.val_paths[:100], self.val_labels[:100])
        
        self.train_dataset = self._build_dataset(self.train_paths, self.train_labels)
        self.val_dataset = self._build_dataset(self.val_paths, self.val_labels)
    
    
    def _load_image(self, imagePath, label):
        image =tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        image= tf.image.resize(image, self.image_size)
        image= tf.cast(image, tf.float32)/255.0 ### Making it O and 1
        return image, label
    
    def _augment_image(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.05)
        return image
    
    def _split_paths_data(self):
        train_paths, val_paths, train_labels, val_labels = train_test_split(self.image_paths, self.labels, 
                                                                            test_size=self.validation_split, stratify=self.labels, random_state=42)
        return train_paths, val_paths, train_labels, val_labels 
    
    def _build_dataset(self, img_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        dataset = dataset.map(self._load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.augument:
            dataset = dataset.map(lambda x, y: (self._augment_image(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.is_shuffle:
            dataset = dataset.shuffle(buffer_size=2024)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    # def 

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
    DataLoader=ImageDataLoader(train_images_path, labels,batch_size=32, image_size=(64,64))
    train_dataset= DataLoader.get_train_dataset()
    val_dataset =DataLoader.get_val_dataset()


    print("length of train datasets = ",len(train_dataset)*32)
    print("length of val datasets = ", len(val_dataset)*32)

    fig, ax =plt.subplots(10,10, figsize=(10,10), sharex=True)
    ax=ax.flatten()
    count=0
    num_zeros=0
    num_ones =0
    for index,(x_train, y_train) in enumerate(train_dataset):
        num_zeros += tf.reduce_sum(tf.cast(tf.equal( y_train, 0), tf.int32)).numpy()
        num_ones += tf.reduce_sum(tf.cast(tf.equal(y_train, 1), tf.int32)).numpy()
        # print(index)

        if index%25==0:
            ax[count].imshow(x_train[0,...].numpy())
            ax[count].axis('off') 
            ax[count].set_title(int(y_train[0].numpy()))
            # print("label:", int(y_train[0].numpy()))
            count+=1
            assert x_train.shape==(32,64,64,3)
    plt.savefig('pictures/data_image.png')
    print("Number of zeros:", num_zeros/(len(train_dataset)*32))
    print("Number of ones:", num_ones/(len(train_dataset)*32))
    train_data_bar=[num_zeros/(len(train_dataset)*32), num_ones/(len(train_dataset)*32)]
    print("validation --------")
    num_zeros=0
    num_ones =0
    for index,(x_val, y_val) in enumerate(val_dataset):
        num_zeros += tf.reduce_sum(tf.cast(tf.equal( y_train, 0), tf.int32)).numpy()
        num_ones += tf.reduce_sum(tf.cast(tf.equal(y_train, 1), tf.int32)).numpy()
        # print(index)
   
        if index==len(val_dataset)-1:
            print(x_val.shape)
        assert x_val.shape==(32,64,64,3)
    print("Number of zeros:", num_zeros/(len(val_dataset)*32))
    print("Number of ones:", num_ones/(len(val_dataset)*32))
    val_data_bar=[num_zeros/(len(val_dataset)*32), num_ones/(len(val_dataset)*32)]
    barWidth = 0.25
    br1 = np.arange(len(val_data_bar)) 
    br2 = [x + barWidth for x in br1] 
   

    
    fig = plt.subplots(figsize =(5, 4)) 
    plt.bar(br1, val_data_bar, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Val')
    plt.bar(br2,train_data_bar , color ='b', width = barWidth, 
        edgecolor ='grey', label ='Train')
    plt.xlabel('Datasets', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Probability', fontweight ='bold', fontsize = 15) 

    plt.xticks([r + barWidth for r in range(len(br1))], 
        ['0', '1'])
    plt.legend()
    plt.savefig('pictures/data_dist.png')


    



