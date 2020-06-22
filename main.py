import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.applications.vgg16 import preprocess_input

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
task = 'which_pose' #'which_pose'
mode = 'train'
env = 'mac'
model_type = 'vgg16' #mobilenet, vgg16, resnet_50, xception, inception_v3
dataset_type = 'neokt' #socar

print('>> Task: ', task)
print('>> Model Type: ', model_type)
print('>> Dataset: ', dataset_type)

if dataset_type == 'socar':

    if env == 'ubuntu':
        base_path = '../'

    else:
        base_path = '/Users/kp/Desktop/work/scratch_detection/socar_dataset'

    img_dir_path = os.path.join(base_path, 'damaged_car_images/')
    label_dir_path = os.path.join(base_path, 'bbox_labels/')

elif dataset_type == 'neokt':
    
    if env == 'ubuntu':
        base_path = '../car-damage-dataset/'

    else:
        base_path = '/Users/kp/Desktop/work/scratch_detection/car-damage-dataset'
    
    if task == 'is_damage':
        img_dir_path = os.path.join(base_path, 'is_damage_dataset')

    elif task == 'severity':
        img_dir_path = os.path.join(base_path, 'damage_severity_dataset')

    elif task == 'which_pose':
        img_dir_path = os.path.join(base_path, 'pose_dataset')


# img - label pair checker
def _dataset_pair_checker(img_dir_path, label_dir_path):
    
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
        img = cv2.imread(os.path.join(img_dir_path, img_file_name))[:,:,::-1] #read as RGB
        img = img.astype(np.float32)
        img = preprocess_input(img)

        #image resizing
        resized_img = cv2.resize(img, dsize=reshape_size, interpolation=cv2.INTER_CUBIC)
        
        img_container.append(resized_img)

        if idx % 200 == 0:
            print('>> Processed Images: ', str(idx))
        idx = idx + 1

    img_container = np.asarray(img_container)

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
if dataset_type == 'socar':
    images = read_img(img_dir_path)
    labels = read_label(label_dir_path, 'is_damage')

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

elif dataset_type == 'neokt':
    
    if task == 'is_damage':
        train_normal_path = os.path.join(img_dir_path, 'training', '01-whole')
        train_damage_path = os.path.join(img_dir_path, 'training', '00-damage')
        test_normal_path = os.path.join(img_dir_path, 'validation', '01-whole')
        test_damage_path = os.path.join(img_dir_path, 'validation', '00-damage')

        train_normal_img = read_img(train_normal_path)
        train_damage_img = read_img(train_damage_path)
        test_normal_img = read_img(train_normal_path)
        test_damage_img = read_img(train_damage_path)

        train_normal_label = np.zeros(len(train_normal_img))
        train_damage_label = np.ones(len(train_damage_img))
        test_normal_label = np.zeros(len(test_normal_img))
        test_damage_label = np.ones(len(test_damage_img))

        print(len(train_normal_label))
        print(len(train_damage_label))
        print(len(test_normal_label))
        print(len(test_damage_label))

        train_x = np.concatenate((train_normal_img, train_damage_img), axis=0)
        train_y = np.concatenate((train_normal_label, train_damage_label), axis=0)
        test_x = np.concatenate((test_normal_img, test_damage_img), axis=0)
        test_y = np.concatenate((test_normal_label, test_damage_label), axis=0)

        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y,
            test_size = 0.2,
            random_state = 40
        )

    elif task == 'which_pose':
        train_front_path = os.path.join(img_dir_path, 'training', '00-front')
        train_rear_path = os.path.join(img_dir_path, 'training', '01-rear')
        train_side_path = os.path.join(img_dir_path, 'training', '02-side')

        test_front_path = os.path.join(img_dir_path, 'validation', '00-front')
        test_rear_path = os.path.join(img_dir_path, 'validation', '01-rear')
        test_side_path = os.path.join(img_dir_path, 'validation', '02-side')

        train_front_img = read_img(train_front_path)
        train_rear_img = read_img(train_rear_path)
        train_side_img = read_img(train_side_path)
        
        test_front_img = read_img(test_front_path)
        test_rear_img = read_img(test_rear_path)
        test_side_img = read_img(test_side_path)

        train_front_label = np.full((len(train_front_img), ), 0) # front
        train_rear_label = np.full((len(train_rear_img), ), 1) #rear
        train_side_label = np.full((len(train_side_img), ), 2) #side
        test_front_label = np.full((len(test_front_img), ), 0) # front
        test_rear_label = np.full((len(test_rear_img, ), ), 1) #rear
        test_side_label = np.full((len(test_side_img, ), ), 2) #side

        print(len(train_front_label))
        print(len(train_rear_label))
        print(len(train_side_label))
        print(len(test_front_label))
        print(len(test_rear_label))
        print(len(test_side_label))

        train_x = np.concatenate((train_front_img, train_rear_img, train_side_img), axis=0)
        train_y = np.concatenate((train_front_label, train_rear_label, train_side_label), axis=0)

        test_x = np.concatenate((test_front_img, test_rear_img, test_side_img), axis=0)
        test_y = np.concatenate((test_front_label, test_rear_label, test_side_label), axis=0)

        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y,
            test_size = 0.2,
            random_state = 40
        )

print('train_x: ', train_x.shape)
print('val_x: ', val_x.shape)
print('test_x: ', test_x.shape)
print('train_y: ', train_y.shape)
print('val_y: ', val_y.shape)
print('test_y: ', test_y.shape)

#1. Train Mode?
if mode == 'train':

    if task == 'is_damage':
        classifier = transfer_learning_model(
            train_x, train_y, val_x, val_y, test_x, test_y,
            num_class = 2,
            epoch=10,
            batch_size=100,
            model_type=model_type, #vgg_16, resnet_50, xception, inception_v3, mobilenet_v2
            reshape_size=reshape_size,
            l1_weight = 0.00001,
            l2_weight = 0.00001
        )

    elif task == 'which_pose':

        classifier = transfer_learning_model(
            train_x, train_y, val_x, val_y, test_x, test_y,
            num_class = 3,
            epoch=10,
            batch_size=100,
            model_type=model_type, #vgg_16, resnet_50, xception, inception_v3, mobilenet_v2
            reshape_size=reshape_size,
            l1_weight = 0.00001,
            l2_weight = 0.00001
        )


#2. Inference Mode?



