import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import *
from keras.losses import *
from keras.optimizers import *
from keras.activations import *
from keras.layers import Conv2D, MaxPooling2D
import time
import datetime

# Adapt to computer
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
log_dir = "../log_furnitures/"
model_dir = "../models/"

model_desc="2018-04-25_09:03:00_R128_P1_mse-sgd_C2D-16(3, 3)R-MP2D(2, 2)-D-128-Sm.0"

batch_size = 1400

# Dataset
nb_train_samples = 191129
nb_validation_samples = 6289
img_width, img_height = 128, 128
input_shape = (img_width,img_height,3)

# Learning
epochs = 10

model = load_model(model_dir + model_desc)

tb_callback = TensorBoard(log_dir + model_desc)
model_checkpoint = ModelCheckpoint(model_dir + model_desc)

callback_list = [tb_callback, model_checkpoint]

# Generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callback_list,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)