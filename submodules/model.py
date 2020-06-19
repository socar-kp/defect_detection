import keras

from keras import models, layers
from keras.applications import VGG16, ResNet50, Xception, InceptionV3, MobileNet
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.layers import BatchNormalization, Activation

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

# to download pretrained models with adequate authorization
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def transfer_learning_model(train_x, train_y, val_x, val_y, test_x, test_y, num_class, epoch, batch_size, model_type, reshape_size, l1_weight, l2_weight):

    print(train_x.shape)
    print(type(train_x))
    print('\n')
    print(train_x[0].shape)

    # change label into one hot
    train_y = keras.utils.to_categorical(train_y, num_classes=num_class)
    val_y = keras.utils.to_categorical(val_y, num_classes=num_class)
    #test_y = keras.utils.to_categorical(test_y, num_classes=num_class)
    print(test_y)

    if model_type == 'vgg16':
        pre_trained = VGG16(
            weights='imagenet',
            include_top=False, # True if we want to add Fully Connected Layer at the Last (False)
            input_shape=reshape_size + (3,)
        )
        pre_trained.trainable = False  # False if we want to freeze the weight

    elif model_type == 'resnet_50':
        pre_trained = ResNet50(
            weights='imagenet'
        )

    elif model_type == 'xception':
        pre_trained = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=reshape_size + (3,)
        )

    elif model_type == 'inception_v3':
        pre_trained = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=reshape_size + (3,)
        )

    elif model_type == 'mobilenet':
        pre_trained = MobileNet(
            weights='imagenet',
            include_top=False,
            input_shape=reshape_size + (3,)
        )

    #pre_trained.summary()
    # Add Fine-Tuning Layers
    finetune_model = models.Sequential()
    finetune_model.add(pre_trained)
    
    if model_type == 'resnet_50':
        pass

    else:
        finetune_model.add(layers.Flatten())

    finetune_model.add(layers.Dense(num_class*128,# activation='relu',
        kernel_regularizer=regularizers.l1_l2(
            l1=l1_weight,
            l2=l2_weight)
        ))
    finetune_model.add(BatchNormalization())
    finetune_model.add(Activation('relu'))
    #finetune_model.add(layers.Dense(num_class*64, activation='relu'))
    
    finetune_model.add(layers.Dense(num_class*32,# activation='relu',
        kernel_regularizer=regularizers.l1_l2(
                l1=l1_weight,
                l2=l2_weight)
        ))
    finetune_model.add(BatchNormalization())
    finetune_model.add(Activation('relu'))
    
    #finetune_model.add(layers.Dense(num_class*16, activation='relu'))
    
    finetune_model.add(layers.Dense(num_class*8,# activation='relu',
        kernel_regularizer=regularizers.l1_l2(
                    l1=l1_weight,
                    l2=l2_weight)    
        ))
    finetune_model.add(BatchNormalization())
    finetune_model.add(Activation('relu'))
    
    finetune_model.add(layers.Dense(num_class, activation='softmax')) # Final Activation

    #finetune_model.summary()

    # Compile
    finetune_model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics=['acc']
    )

    history = finetune_model.fit(
        train_x,
        train_y,
        epochs=epoch,
        batch_size = batch_size,
        validation_data = (val_x, val_y)
    )

    # Test Performance
    '''
    TODO: Result 해결하는데 이슈가 있음 ### !
    '''
    y_pred = finetune_model.predict(test_x) #np.argmax
    y_pred = np.argmax(y_pred, axis=1)
    print('>> Predicted Results')
    print(y_pred)
    
    test_y = np.argmax(test_y, axis=1)
    print('>> Ground Truth')
    print(test_y)

    accuracy = accuracy_score(test_y, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_y, y_pred, average='micro')
    
    print(">> Test Performance <<")
    print('Acc: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1_score)
    