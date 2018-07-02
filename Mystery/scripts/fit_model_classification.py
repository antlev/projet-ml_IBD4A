from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import *
import time
import datetime
import numpy as np
from scripts.my_classes import MysterySequencer, all_diff_element

# Experiment name
ts = time.time()
date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

experiment_name = "MYST_BC_RM_LINEAR" + date_time
print("Launching experiment : " + experiment_name)

# Adapt to computer
input_data_path = 'D://data//2018_04_28_full_train-000000-input.npy'
output_classification_path = 'D://data//2018_04_28_full_train-000000-output1.npy'
log_dir = "logs/"
model_dir = "models/"

RANDOM_SEED = 42
INPUT_SIZE = 15444000
NUMBER_OF_CLASSES = 1020
BATCH_SIZE = 150000
INPUT_SHAPE=(1020,)
# Learning
EPOCHS = 50

# Load data
input_data=np.load(input_data_path, mmap_mode='r')
output_classification=np.load(output_classification_path, mmap_mode='r')

# Before preprocessing
print("Raw data :")
print("input_data shape : " + str(input_data.shape) + " - input_data[0] size : " + str(input_data[0].size) + " - different values are : " + str(all_diff_element(input_data[0], True)))
print("output_data shape : " + str(output_classification.shape) + " - different values are : " + str(all_diff_element(output_classification, False)))

# Split train validation
input_data_train, input_data_validation = np.split(input_data, [int(.7*len(input_data))])
output_classification_train, output_classification_validation = np.split(output_classification, [int(.7*len(output_classification))])

# After preprocessing
print("After split :")
print("input_data_train shape :" + str(input_data_train.shape) + " - input_data_validation shape :" + str(input_data_validation.shape) )
print("output_classification_train shape :" + str(output_classification_train.shape) + " - output_classification_validation shape :" + str(output_classification_validation.shape))

# Generator
train_generator = MysterySequencer(input_data_train, output_classification_train, NUMBER_OF_CLASSES, BATCH_SIZE, True)
val_generator = MysterySequencer(input_data_validation, output_classification_validation, NUMBER_OF_CLASSES, BATCH_SIZE, True)

# Model
model = Sequential()
model.add(Dense(1020, activation='softmax', input_shape=INPUT_SHAPE))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Callback
tb_callback = TensorBoard(log_dir + experiment_name)
model_checkpoint = ModelCheckpoint(model_dir + experiment_name)
callback_list = [tb_callback, model_checkpoint]

# Fit model
model.fit_generator(
    train_generator,
    steps_per_epoch=INPUT_SIZE // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callback_list,
    validation_data=val_generator)