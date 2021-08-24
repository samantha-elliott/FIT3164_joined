# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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
# -

# Check to see if TensorFlow recognises my GPU
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices()) 

# ## Load the images

# Load the train and test data sets
test = pd.read_csv("./data/csv/calc_case_description_test_set.csv")
train = pd.read_csv("./data/csv/calc_case_description_train_set.csv")

# +
# Taken from https://github.com/BiditPakrashi/CNN-Xray/blob/master/Working_With_Chest_XRay_Images.ipynb
# https://towardsdatascience.com/deep-learning-with-x-ray-eyes-eae0ac39b85f

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0,
    height_shift_range=0,
    vertical_flip=False,)

def preprocess_img(img, mode):
    img = (img - img.min())/(img.max() - img.min())
    img = rescale(img, 0.25, multichannel=True, mode='constant')
    if mode == 'train':
        if np.random.randn() > 0:
            img = datagen.random_transform(img)
    return img



# -

class PreprocessImages:
    def __init__(self, width, height, inter= cv2.INTER_AREA):
        self.width= width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        try:
            image= cv2.resize(image, (self.width, self.height), interpolation = self.inter)
            b,g,r= cv2.split(image)
            image= cv2.merge((r,g,b))
            return image
        except Exception:
            pass
        
    def load_images(self, df, file_path, label_column, file_path_column):
        data = []
        labels = []
        for i in tqdm(range(len(df))):
            path = df[file_path_column][i].split("/")[2]
            label = df[label_column][i]
            for image in os.listdir(file_path + path):
                img = cv2.imread(file_path + path + "/" + image)
                img = self.preprocess(img)
                data.append(img)
                labels.append(label)
        return data, labels


images = PreprocessImages(256, 256)
train_data, train_labels = images.load_images(train, "./data/jpeg/", "pathology", "image file path")

test_data, test_labels = images.load_images(test, "./data/jpeg/", "pathology", "image file path")

# ## Pre-process and split images

# +
# Pre-process the labels
le = LabelEncoder()

x_train = np.array(train_data)
y_train = np.array(train_labels)
# x_train = np.expand_dims(x_train, axis=-1)
le.fit(list(set(y_train)))
y_train = le.transform(y_train)

x_test = np.array(test_data) 
# x_test = np.expand_dims(x_test, axis=-1)

y_test = np.array(test_labels)
y_test = le.transform(y_test)
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
# -

# ## Build the ResNet model

# +
import tensorflow as tf 

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

X = tf.random.uniform(shape=(1, 256, 256, 3))
for layer in create_ResNet([256,256,3]).layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
# -

model = create_ResNet(num_units=3)
model.compile(optimizer= tf.keras.optimizers.Adam(0.001), loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 64, epochs=100, validation_data=(X_val, y_val))
model.evaluate(x_test, y_test)

mirrored_strategy = tf.distribute.MirroredStrategy()
#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", ":/gpu:2", ":gpu:3"])
def reset_keras():
    tf.keras.backend.clear_session
reset_keras()


model.save("ResNet50.h5")
saved_model  = tf.keras.models.load_model('ResNet50.h5')
# saved_model.fit(X_train, y_train, batch_size = 64, epochs= 100, validation_data=(X_val, y_val))
saved_model.evaluate(x_test, y_test)