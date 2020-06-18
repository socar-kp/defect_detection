import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import os
import json
import cv2

from sklearn.model_selection import train_test_split
from submodules.model import *

'''
1. Read Dataset
* number of imgs = 2000 (.DS_Store --> erased)
* number of labels = 2000
'''

reshape_size = (224,224)
task = 'is_damage' #'which_pose'
mode = 'train'
env = 'mac'
model_type = 'inception_v3' #vgg_16, resnet_50, xception, inception_v3

img_dir_path = '/Users/kp/Desktop/work/scratch_detection/socar_dataset/damaged_car_images/' #path of the image
label_dir_path = '/Users/kp/Desktop/work/scratch_detection/socar_dataset/bbox_labels/' #path of the label

if env == 'ubuntu':
    img_dir_path = './damaged_car_images/' #path of the image
    label_dir_path = './bbox_labels/' #path of the label

# img - label pair checker
def dataset_pair_checker(img_dir_path, label_dir_path):
    
    error_img_container = list() # A container which holds the name of unmatched img-label

    for img_file_name in os.listdir(img_dir_path):
        img_name = img_file_name[:-4] #drop .jpg

        if img_name + '.json' in os.listdir(label_dir_path):
            pass
        
        else:
            print('>> Error Found on ', img_file_name, '!')
            error_img_container.append(img_file_name)

    if len(error_img_container) != 0:
        return False
        
    else:
        return True

# read img 
def read_img(img_dir_path):
    
    img_container = list()
    img_file_names = os.listdir(img_dir_path)

    idx = 0
    for img_file_name in img_file_names:
        
        #read image
        img = cv2.imread(os.path.join(img_dir_path, img_file_name))[:,:,::-1]/255 #read as RGB

        #image resizing
        resized_img = cv2.resize(img, dsize=reshape_size, interpolation=cv2.INTER_CUBIC)

        img_container.append(resized_img)

        if idx % 200 == 0:
            print('>> Processed Images: ', str(idx))
        idx = idx + 1

    img_container = np.asarray(img_container)
    #img_container = np.concatenate(img_container)
    print(img_container.shape)

    return img_container

def read_label(label_dir_path, task):

    label_container = list()

    label_file_names = os.listdir(label_dir_path)

    for label_file_name in label_file_names: # iterate every files

        with open(os.path.join(label_dir_path, label_file_name), 'r') as f:
            label_dict = json.load(f)

            if task == 'is_damage':
                label = label_dict['damage']
                label = int(label)

                label_container.append(label)

            elif task == 'which_pose':
                label = label_dict['picture_position'] # label in str
                label = int(label) # transform str label into int

                label_container.append(label) # append label into the container

            ### TO-DO: Add additional label parsing conditions            
            else:
                print('>> No Labels are parsed!')
    
    label_container = np.asarray(label_container)
    print(np.unique(label_container))    
    print(label_container.shape)
    return label_container

# 0. read image and labels
images = read_img(img_dir_path)
labels = read_label(label_dir_path, is_damage)
print('>> Images are: ', images.shape)
print('>> Labels are: ', labels.shape)

train_x, test_x, train_y, test_y = train_test_split(
    images, labels,
    test_size = 0.2,
    random_state = 40
)

train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y,
    test_size = 0.1,
    random_state = 40
)

print('train_x: ', train_x.shape)
print('val_x: ', val_x.shape)
print('test_x: ', test_x.shape)

#1. Train Mode?
if mode == 'train':

    if task == 'is_damage':
        classifier = transfer_learning_model(
            train_x, train_y, val_x, val_y, test_x, test_y,
            num_class = 2,
            epoch=10,
            batch_size=100,
            model_type=model_type, #vgg_16, resnet_50, xception, inception_v3, mobilenet_v2
            reshape_size=reshape_size
        )

    elif task == 'which_pose':
        classifier = transfer_learning_model(
            train_x, train_y, val_x, val_y, test_x, test_y,
            num_class = 6,
            epoch=10,
            batch_size=100,
            model_type=model_type, #vgg_16, resnet_50, xception, inception_v3, mobilenet_v2
            reshape_size=reshape_size
        )




#2. Inference Mode?



