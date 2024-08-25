# Train_1=TrainClassifier(optimizer, "accuracy", _image_size=(224,224), _batch_size=32, epochs=5)()
 




    # BCe=tf.keras.losses.BinaryCrossentropy(from_logits=True)


    # train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    # val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()



    # for step, (x_train, y_true) in enumerate(train_dataset):
    #     logits = model(x_train)


        # loss_value=binary_cross_entropy(y_true, y_pred=logits)

    ## Loss verification
    # for index, (x_train, y_true) in enumerate(train_dataset):
    #     y_pred = model(x_train)
    #     # difference = y_train - y_pred
    #     # # print(difference  )
    #     print(BCe(y_true, y_pred))
        
    #     print(binary_cross_entropy(y_true, y_pred))
    #     print("----")
    #     if index==5:
    #         break
        
    # y_true= tf.constant([[0., 0., 0., 0., 1., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]])
    # y_pred= tf.constant([[ 1.8055,  0.9193, -0.2527,  1.0489,  0.5396, -1.2046, -0.9479,  0.8274,-0.0548, -0.1902],[ 1.8055,  0.9193, -0.2527,  1.0489,  0.5396, -1.2046, -0.9479,  0.8274,-0.0548, -0.1902]])
    # print(BCe(y_true, y_pred))
    # print(binary_cross_entropy(y_true, y_pred))
    # model.compile(optimizer='adam' ,loss='binary_crossentropy' , metrics=['acc'])
    
    # history = model.fit(train_dataset, epochs=3)
    # print(model.summary())
    
    # opt =tf.keras.optimizers.Adam(learning_rate=1e-5)
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


    # history = model.fit(train_dataset,steps_per_epoch=len(train_dataset),epochs=10,validation_data=val_dataset,validation_steps=len(val_dataset))

    #  #### training whole model
    # train_images_path=read.getTrainDataPathList( "data/train_img")
    # labels=read.getTrainLabels("data/label_train.txt")
    # Datasets_init=dataLoader.ImageDataLoader(train_images_path, labels, batch_size=8,image_size=(224,224))
    # train_dataset= Datasets_init.get_train_dataset()
    # val_dataset =  Datasets_init.get_val_dataset()
    
    # ## Testing phase
    # ### model
    # model = models.vg16.VGG16_Based(16,input_shape=(224,224,3))
    # optimizer =tf.keras.optimizers.Adam(learning_rate=1e-5)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # history = model.fit(train_dataset,steps_per_epoch=len(train_dataset),epochs=30,validation_data=val_dataset,validation_steps=len(val_dataset))

    ###### losss function try##############################################
    ### lossfunction try
    # y_true = tf.convert_to_tensor([[0.0, 1.0], [0.0, 0.0]])
    # y_pred = tf.convert_to_tensor([[-18.6, 0.51], [2.94, -12.8]])

    # ##since I was not having sigmoid then I use it
    # @tf.function
    # def bce(y_true, y_pred,weights=[0.2,0.8]):
    #     bce_1 = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #     sample_weight_ = tf.where(tf.equal(y_true, 0), weights[0], weights[1])
        
    #     return tf.reduce_mean(bce_1(y_true, y_pred)*sample_weight_)
    # ### true means we will apply sigmod function
    # print(bce(y_true, y_pred,weights=[1.0,1.0]))
    # print(binary_cross_entropy(y_true, tf.math.sigmoid(y_pred),epsilon_=1e-7))

    
    # print(bce(y_true, y_pred,weights=[0.2,0.8]))
    # print(weighted_binary_cross_entropy(y_true, tf.math.sigmoid(y_pred),epsilon_=1e-7, weights=[0.2, 0.8]))

    # print( weighted_binary_cross_entropy_2(y_true,tf.math.sigmoid( y_pred), weights=[0.2, 0.8]))

    # print(tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)))