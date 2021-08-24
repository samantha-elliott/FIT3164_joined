"""
Model classes to be imported so there are not issues with loading the saved model
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Back, Style
sns.set(style='dark')

from skimage.transform import rescale
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.layers import MaxPooling2D, Input, Lambda, concatenate, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Activation, add, BatchNormalization, LeakyReLU, DepthwiseConv2D
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.num_channels = num_channels

    def call(self, X):
#         Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.nn.leaky_relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
#         return tf.nn.leaky_relu(Y, alpha=0.01)
        return tf.keras.activations.relu(Y, alpha=0.01)

@tf.keras.utils.register_keras_serializable()
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.num_residuals = num_residuals
        self.first_block = first_block
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def get_config(self):
        config = super(ResnetBlock, self).get_config()
        config.update({"num_channels": self.num_channels,
                      "num_residuals": self.num_residuals,
                      "first_block": self.first_block})
        return config

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X

@tf.keras.utils.register_keras_serializable()
def create_ResNet(input_shape=[256,256,3], num_units=3):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', kernel_initializer='he_normal', input_shape = [256,256,3]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        ResnetBlock(1024, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=num_units, activation= 'softmax')])