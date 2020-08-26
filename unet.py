#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import warnings
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split


import skimage.io                                     
import skimage.transform  

import tensorflow.compat.v1 as tf

BATCH_SIZE = 15 
IMG_WIDTH = 256 
IMG_HEIGHT = 256 
IMG_CHANNELS = 3
TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage1_test/'

seed = 42


train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
np.random.seed(10)

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


# Kinda hard to figure out, but this function gets the training data
def get_training_data(path, output_shape=(None, None)):
    img_paths = ['{0}/{1}/images/{1}.png'.format(path, id) for id in os.listdir(path)]
    X_data = np.array([skimage.transform.resize(skimage.io.imread(path)[:,:,:3], output_shape=output_shape, mode='constant', preserve_range=True) for path in img_paths], dtype=np.uint8) 
    return X_data

X_train = get_training_data(TRAIN_PATH, output_shape=(IMG_HEIGHT,IMG_WIDTH))


# What do you think this does? I'm thinking it gets the training masks
def get_training_masks(path, output_shape=(None, None)):
    img_paths = [glob.glob('{0}/{1}/masks/*.png'.format(path, id)) for id in os.listdir(path)]
    Y_data = []
    for i, img_masks in tqdm(enumerate(img_paths)):  #Ok this part is a little more complicated we have to combine all the masks for each image into one
        masks = skimage.io.imread_collection(img_masks).concatenate() 
        mask = np.max(masks, axis=0)                                   
        mask = skimage.transform.resize(mask, output_shape=output_shape+(1,), mode='constant', preserve_range=True)
        Y_data.append(mask)
    Y_data = np.array(Y_data, dtype=np.bool)
    
    return Y_data

Y_train = get_training_masks(TRAIN_PATH, output_shape=(IMG_HEIGHT,IMG_WIDTH))


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.9, test_size=0.1, random_state=seed)

# Creating the training Image and Mask generator
image_datagen = image.ImageDataGenerator(shear_range=0.2, rotation_range=50, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.2, fill_mode='reflect')
mask_datagen = image.ImageDataGenerator(shear_range=0.2, rotation_range=50, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.2, fill_mode='reflect')

#Fit the data to the 
image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(Y_train, augment=True, seed=seed)

xtrain=image_datagen.flow(X_train,batch_size=BATCH_SIZE,shuffle=True, seed=seed)
ytrain=mask_datagen.flow(Y_train,batch_size=BATCH_SIZE,shuffle=True, seed=seed)

X_datagen_val = image.ImageDataGenerator()
Y_datagen_val = image.ImageDataGenerator()
X_datagen_val.fit(X_test, augment=True, seed=seed)
Y_datagen_val.fit(Y_test, augment=True, seed=seed)

xtest = X_datagen_val.flow(X_test, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
ytest = Y_datagen_val.flow(Y_test, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    
    
# combine generators into one which yields image and masks
train_generator = zip(xtrain, ytrain)
val_generator = zip(xtest, ytest)


#How accuracy is measured
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        tf.compat.v1.Session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)




#Generic Unet Structure
def get_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    i = Lambda(lambda x: x / 255) (inputs)

    #Down sampling for encoder
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (i)
    conv1 = Dropout(0.1) (conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    pool1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool1)
    conv2 = Dropout(0.1) (conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    pool2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool2)
    conv3 = Dropout(0.2) (conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    pool3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool3)
    conv4 = Dropout(0.2) (conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pool4)
    conv5 = Dropout(0.3) (conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv5)


    #Upsampling for decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
    u6 = concatenate([u6, conv4])
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    conv6 = Dropout(0.2) (conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
    u7 = concatenate([u7, conv3])
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    conv7 = Dropout(0.2) (conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
    u8 = concatenate([u8, conv2])
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    conv8 = Dropout(0.1) (conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
    u9 = concatenate([u9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    conv9 = Dropout(0.1) (conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    return model


#Train the network Actual line for training is commented out so results can be seen without training an entire model

model = get_unet()
#results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=250, epochs=3)

model = load_model('unet.h5', custom_objects={'mean_iou': mean_iou})

ptrain = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
ptrainval = (ptrain > 0.5).astype(np.uint8)

#Show classifier working on testing data
for i in range(5):
    ix = random.randint(0, len(ptrainval))
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(ptrainval[ix]))
    plt.show()


    

