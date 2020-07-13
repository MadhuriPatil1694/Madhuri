
from __future__ import print_function
import numpy as np
import keras
from keras.preprocessing.image import img_to_array, array_to_img, ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers.core import Layer
from keras import backend as K

import tensorflow as tf

from keras.utils.conv_utils import convert_kernel

# Loading Datasets

from keras.datasets import mnist,cifar10




# Choose dataset (mnist or cifar10)


# For MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
data_augmentation = False

train_images = np.reshape(train_images,(train_images.shape[0], 28, 28,1))
test_images = np.reshape(test_images,(test_images.shape[0],28, 28,1))

	
# Uncomment for using CIFAR-10
# For CIFAR-10 set
#data_augmentation = True
#(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# Ste to True for enabling mean substract
subtract_pixel_mean = False

# Converting labets to OneHot
train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)

# Resize images to 224 for input
train_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in train_images])

test_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((224,224))) for im in test_images])




# Normalizaton
train_images = train_images/ 255.0
test_images = test_images / 255.0

if subtract_pixel_mean:
    x_train_mean = np.mean(train_images, axis=0)
    train_images -= x_train_mean
    test_images -= x_train_mean



print(train_labels.shape)

input = Input(shape=(224, 224,train_images.shape[-1]))


class LRN_layer(Layer):

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN_layer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b,  r, c, ch = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = K.square(x) # square the input
      
        input_sqr = tf.pad(input_sqr, [[0, 0],  [0, 0], [0, 0], [half_n, half_n]])
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:,  :, :,i:i+ch]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PoolHelper_Layer(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper_Layer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:,1:,1:,:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





#input = Input(shape=(224, 224,1))

input_pad = ZeroPadding2D(padding=(3, 3))(input)
conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu', name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(input)
conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)    
pool1_helper = PoolHelper_Layer()(conv1_zero_pad)
pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool1/3x3_s2')(pool1_helper)    
pool1_norm1 = LRN_layer(name='pool1/norm1')(pool1_3x3_s2)
#print(pool1_norm1.shape)

conv2_3x3_reduce = Conv2D(64, (1,1), padding='same', activation='relu', name='conv2/3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_norm1)
conv2_3x3 = Conv2D(192, (3,3), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
conv2_norm2 = LRN_layer(name='conv2/norm2')(conv2_3x3)
conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)  
pool2_helper = PoolHelper_Layer()(conv2_zero_pad)
pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool2/3x3_s2')(pool2_helper)

incept_layer_3a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='incept_layer_3a/1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
incept_layer_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='incept_layer_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
incept_layer_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_3a_3x3_reduce)
incept_layer_3a_3x3 = Conv2D(128, (3,3), padding='valid', activation='relu', name='incept_layer_3a/3x3', kernel_regularizer=l2(0.0002))(incept_layer_3a_3x3_pad)
incept_layer_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='incept_layer_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
incept_layer_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_3a_5x5_reduce)
incept_layer_3a_5x5 = Conv2D(32, (5,5), padding='valid', activation='relu', name='incept_layer_3a/5x5', kernel_regularizer=l2(0.0002))(incept_layer_3a_5x5_pad)
incept_layer_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_3a/pool')(pool2_3x3_s2)
incept_layer_3a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='incept_layer_3a/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_3a_pool)
incept_layer_3a_output = Concatenate(axis=-1, name='incept_layer_3a/output')([incept_layer_3a_1x1,incept_layer_3a_3x3,incept_layer_3a_5x5,incept_layer_3a_pool_proj])

incept_layer_3b_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='incept_layer_3b/1x1', kernel_regularizer=l2(0.0002))(incept_layer_3a_output)
incept_layer_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='incept_layer_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(incept_layer_3a_output)
incept_layer_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_3b_3x3_reduce)
incept_layer_3b_3x3 = Conv2D(192, (3,3), padding='valid', activation='relu', name='incept_layer_3b/3x3', kernel_regularizer=l2(0.0002))(incept_layer_3b_3x3_pad)
incept_layer_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='incept_layer_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(incept_layer_3a_output)
incept_layer_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_3b_5x5_reduce)
incept_layer_3b_5x5 = Conv2D(96, (5,5), padding='valid', activation='relu', name='incept_layer_3b/5x5', kernel_regularizer=l2(0.0002))(incept_layer_3b_5x5_pad)
incept_layer_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_3b/pool')(incept_layer_3a_output)
incept_layer_3b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='incept_layer_3b/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_3b_pool)
incept_layer_3b_output = Concatenate(axis=-1, name='incept_layer_3b/output')([incept_layer_3b_1x1,incept_layer_3b_3x3,incept_layer_3b_5x5,incept_layer_3b_pool_proj])

incept_layer_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_3b_output)
pool3_helper = PoolHelper_Layer()(incept_layer_3b_output_zero_pad)
pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool3/3x3_s2')(pool3_helper)

incept_layer_4a_1x1 = Conv2D(192, (1,1), padding='same', activation='relu', name='incept_layer_4a/1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
incept_layer_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='incept_layer_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
incept_layer_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_4a_3x3_reduce)
incept_layer_4a_3x3 = Conv2D(208, (3,3), padding='valid', activation='relu', name='incept_layer_4a/3x3' ,kernel_regularizer=l2(0.0002))(incept_layer_4a_3x3_pad)
incept_layer_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='incept_layer_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
incept_layer_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_4a_5x5_reduce)
incept_layer_4a_5x5 = Conv2D(48, (5,5), padding='valid', activation='relu', name='incept_layer_4a/5x5', kernel_regularizer=l2(0.0002))(incept_layer_4a_5x5_pad)
incept_layer_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_4a/pool')(pool3_3x3_s2)
incept_layer_4a_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='incept_layer_4a/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_4a_pool)
incept_layer_4a_output = Concatenate(axis=-1, name='incept_layer_4a/output')([incept_layer_4a_1x1,incept_layer_4a_3x3,incept_layer_4a_5x5,incept_layer_4a_pool_proj])

loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1/ave_pool',dim_ordering="tf")(incept_layer_4a_output)
loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1/conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
loss1_flat = Flatten()(loss1_conv)
loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)
loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
loss1_classifier = Dense(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
loss1_classifier_act = Activation('softmax')(loss1_classifier)

incept_layer_4b_1x1 = Conv2D(160, (1,1), padding='same', activation='relu', name='incept_layer_4b/1x1', kernel_regularizer=l2(0.0002))(incept_layer_4a_output)
incept_layer_4b_3x3_reduce = Conv2D(112, (1,1), padding='same', activation='relu', name='incept_layer_4b/3x3_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4a_output)
incept_layer_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_4b_3x3_reduce)
incept_layer_4b_3x3 = Conv2D(224, (3,3), padding='valid', activation='relu', name='incept_layer_4b/3x3', kernel_regularizer=l2(0.0002))(incept_layer_4b_3x3_pad)
incept_layer_4b_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='incept_layer_4b/5x5_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4a_output)
incept_layer_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_4b_5x5_reduce)
incept_layer_4b_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='incept_layer_4b/5x5', kernel_regularizer=l2(0.0002))(incept_layer_4b_5x5_pad)
incept_layer_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_4b/pool')(incept_layer_4a_output)
incept_layer_4b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='incept_layer_4b/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_4b_pool)
incept_layer_4b_output = Concatenate(axis=-1, name='incept_layer_4b/output')([incept_layer_4b_1x1,incept_layer_4b_3x3,incept_layer_4b_5x5,incept_layer_4b_pool_proj])

incept_layer_4c_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='incept_layer_4c/1x1', kernel_regularizer=l2(0.0002))(incept_layer_4b_output)
incept_layer_4c_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='incept_layer_4c/3x3_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4b_output)
incept_layer_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_4c_3x3_reduce)
incept_layer_4c_3x3 = Conv2D(256, (3,3), padding='valid', activation='relu', name='incept_layer_4c/3x3', kernel_regularizer=l2(0.0002))(incept_layer_4c_3x3_pad)
incept_layer_4c_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='incept_layer_4c/5x5_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4b_output)
incept_layer_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_4c_5x5_reduce)
incept_layer_4c_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='incept_layer_4c/5x5', kernel_regularizer=l2(0.0002))(incept_layer_4c_5x5_pad)
incept_layer_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_4c/pool')(incept_layer_4b_output)
incept_layer_4c_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='incept_layer_4c/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_4c_pool)
incept_layer_4c_output = Concatenate(axis=-1, name='incept_layer_4c/output')([incept_layer_4c_1x1,incept_layer_4c_3x3,incept_layer_4c_5x5,incept_layer_4c_pool_proj])

incept_layer_4d_1x1 = Conv2D(112, (1,1), padding='same', activation='relu', name='incept_layer_4d/1x1', kernel_regularizer=l2(0.0002))(incept_layer_4c_output)
incept_layer_4d_3x3_reduce = Conv2D(144, (1,1), padding='same', activation='relu', name='incept_layer_4d/3x3_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4c_output)
incept_layer_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_4d_3x3_reduce)
incept_layer_4d_3x3 = Conv2D(288, (3,3), padding='valid', activation='relu', name='incept_layer_4d/3x3', kernel_regularizer=l2(0.0002))(incept_layer_4d_3x3_pad)
incept_layer_4d_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='incept_layer_4d/5x5_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4c_output)
incept_layer_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_4d_5x5_reduce)
incept_layer_4d_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='incept_layer_4d/5x5', kernel_regularizer=l2(0.0002))(incept_layer_4d_5x5_pad)
incept_layer_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_4d/pool')(incept_layer_4c_output)
incept_layer_4d_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='incept_layer_4d/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_4d_pool)
incept_layer_4d_output = Concatenate(axis=-1, name='incept_layer_4d/output')([incept_layer_4d_1x1,incept_layer_4d_3x3,incept_layer_4d_5x5,incept_layer_4d_pool_proj])

loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss2/ave_pool')(incept_layer_4d_output)
loss2_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss2/conv', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
loss2_flat = Flatten()(loss2_conv)
loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)
loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
loss2_classifier = Dense(1000, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)
loss2_classifier_act = Activation('softmax')(loss2_classifier)

incept_layer_4e_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='incept_layer_4e/1x1', kernel_regularizer=l2(0.0002))(incept_layer_4d_output)
incept_layer_4e_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='incept_layer_4e/3x3_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4d_output)
incept_layer_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_4e_3x3_reduce)
incept_layer_4e_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='incept_layer_4e/3x3', kernel_regularizer=l2(0.0002))(incept_layer_4e_3x3_pad)
incept_layer_4e_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='incept_layer_4e/5x5_reduce', kernel_regularizer=l2(0.0002))(incept_layer_4d_output)
incept_layer_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_4e_5x5_reduce)
incept_layer_4e_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='incept_layer_4e/5x5', kernel_regularizer=l2(0.0002))(incept_layer_4e_5x5_pad)
incept_layer_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_4e/pool')(incept_layer_4d_output)
incept_layer_4e_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='incept_layer_4e/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_4e_pool)
incept_layer_4e_output = Concatenate(axis=-1, name='incept_layer_4e/output')([incept_layer_4e_1x1,incept_layer_4e_3x3,incept_layer_4e_5x5,incept_layer_4e_pool_proj])

incept_layer_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_4e_output)
pool4_helper = PoolHelper_Layer()(incept_layer_4e_output_zero_pad)
pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool4/3x3_s2')(pool4_helper)

incept_layer_5a_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='incept_layer_5a/1x1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
incept_layer_5a_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='incept_layer_5a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
incept_layer_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_5a_3x3_reduce)
incept_layer_5a_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='incept_layer_5a/3x3', kernel_regularizer=l2(0.0002))(incept_layer_5a_3x3_pad)
incept_layer_5a_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='incept_layer_5a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
incept_layer_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_5a_5x5_reduce)
incept_layer_5a_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='incept_layer_5a/5x5', kernel_regularizer=l2(0.0002))(incept_layer_5a_5x5_pad)
incept_layer_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_5a/pool')(pool4_3x3_s2)
incept_layer_5a_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='incept_layer_5a/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_5a_pool)
incept_layer_5a_output = Concatenate(axis=-1, name='incept_layer_5a/output')([incept_layer_5a_1x1,incept_layer_5a_3x3,incept_layer_5a_5x5,incept_layer_5a_pool_proj])

incept_layer_5b_1x1 = Conv2D(384, (1,1), padding='same', activation='relu', name='incept_layer_5b/1x1', kernel_regularizer=l2(0.0002))(incept_layer_5a_output)
incept_layer_5b_3x3_reduce = Conv2D(192, (1,1), padding='same', activation='relu', name='incept_layer_5b/3x3_reduce', kernel_regularizer=l2(0.0002))(incept_layer_5a_output)
incept_layer_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(incept_layer_5b_3x3_reduce)
incept_layer_5b_3x3 = Conv2D(384, (3,3), padding='valid', activation='relu', name='incept_layer_5b/3x3', kernel_regularizer=l2(0.0002))(incept_layer_5b_3x3_pad)
incept_layer_5b_5x5_reduce = Conv2D(48, (1,1), padding='same', activation='relu', name='incept_layer_5b/5x5_reduce', kernel_regularizer=l2(0.0002))(incept_layer_5a_output)
incept_layer_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(incept_layer_5b_5x5_reduce)
incept_layer_5b_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='incept_layer_5b/5x5', kernel_regularizer=l2(0.0002))(incept_layer_5b_5x5_pad)
incept_layer_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='incept_layer_5b/pool')(incept_layer_5a_output)
incept_layer_5b_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='incept_layer_5b/pool_proj', kernel_regularizer=l2(0.0002))(incept_layer_5b_pool)
incept_layer_5b_output = Concatenate(axis=-1, name='incept_layer_5b/output')([incept_layer_5b_1x1,incept_layer_5b_3x3,incept_layer_5b_5x5,incept_layer_5b_pool_proj])

pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5/7x7_s2')(incept_layer_5b_output)
loss3_flat = Flatten()(pool5_7x7_s1)
pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
loss3_classifier = Dense(10, name='loss3/classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

googlenet = Model(inputs=input, outputs=loss3_classifier_act) #loss1_classifier_act,loss2_classifier_act,loss3_classifier_act



# convert the convolutional kernels for tensorflow
ops = []
for layer in googlenet.layers:
    if layer.__class__.__name__ == 'Conv2D':
        original_w = K.get_value(layer.kernel)
        converted_w = convert_kernel(original_w)
        ops.append(tf.assign(layer.kernel, converted_w).op)
K.get_session().run(ops)


BatchSize = 32
Epochs = 50

model = googlenet
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

if not data_augmentation:
	history = model.fit(train_images, train_labels, batch_size=BatchSize,epochs=Epochs, validation_data=(test_images,test_labels))

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.2,
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        validation_split=0.0)

   
    datagen.fit(train_images)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(train_images, train_labels,
                                     batch_size= BatchSize),steps_per_epoch=len(train_images)/BatchSize,
                        epochs=Epochs,
                        validation_data=(test_images, test_labels),
                        workers=4)

Results = model.evaluate(test_images,  test_labels, verbose=1)

print('\nTest Loss :', Results[0],'\nTest accuracy:', Results[1]*100, "For batch size", BatchSize)

# loss and accuracy plot

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')


plt.legend()


plt.figure()
plt.title('Training and validation loss for')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')


plt.legend()

plt.show()


