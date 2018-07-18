from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import *
import time
import datetime
import numpy as np
from scripts.my_classes import MysterySequencerOutput1

# Experiment name
ts = time.time()
date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

experiment_name = "OUTPUT1_RM_LINEAR_SM_" + date_time
print("Launching experiment : " + experiment_name)

# Adapt to computer
input_data_path = '../data/2018_04_28_full_train-000000-input.npy'
output1_path = '../data/2018_04_28_full_train-000000-output1.npy'
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
output1=np.load(input_data_path, mmap_mode='r')

# Split train validation
input_data_train, input_data_validation = np.split(input_data, [int(.7*len(input_data))])
output1_train, output1_validation = np.split(output1, [int(.7 * len(output1))])

# Generator
train_generator = MysterySequencerOutput1(input_data_train, output1_train, NUMBER_OF_CLASSES, BATCH_SIZE, False)
val_generator = MysterySequencerOutput1(input_data_validation, output1_validation, NUMBER_OF_CLASSES, BATCH_SIZE, False)

# Model
model = Sequential()
model.add(Dense(1020, activation='softmax', input_shape=INPUT_SHAPE))
model.add(Dense(1, activation='softmax', input_shape=INPUT_SHAPE))

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