from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import *
import time
import datetime
import numpy as np
from scripts.my_classes import MysterySequencer

def all_diff_element(array):
    elements=[]
    for i in range((array.size)):
        if array[i] not in elements:
                elements.append(array[i])
    return elements

# Experiment name
ts = time.time()
date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

experiment_name = "TEST_MYSTERY" + date_time
print("Laucnhing experiment : " + experiment_name)

# Adapt to computer
input_data_path = '../data/2018_04_28_full_train-000000-input.npy'
output_classification_path = '../data/2018_04_28_full_train-000000-output1.npy'
output_lstm_path = '../data/2018_04_28_full_train-000000-output2.npy'
log_dir = "../log_furnitures/"
model_dir = "../models/"

RANDOM_SEED = 42
INPUT_SIZE = 15444000
NUMBER_OF_CLASSES = 1020
BATCH_SIZE = 1400

input_data=np.load(input_data_path, mmap_mode='r')
output_classification=np.load(output_classification_path, mmap_mode='r')
output_lstm=np.load(output_lstm_path, mmap_mode='r')

print("input_data shape : " + str(input_data.shape) + " - different values are : " + str(all_diff_element(input_data[0])))
input_data_train, input_data_validation = np.split(input_data, [int(.7*len(input_data))])
print("input_data_train shape :" + str(input_data_train.shape) + "input_data_validation shape :" + str(input_data_validation.shape) )

output_classification_train, output_classification_validation = np.split(output_classification, [int(.7*len(output_classification))])
print("output_classification_train shape :" + str(output_classification_train.shape) + "output_classification_validation shape :" + str(output_classification_validation.shape))

output_lstm_train, output_lstm_validation = np.split(output_lstm, [int(.7*len(output_lstm))])
print("output_lstm_train shape :" + str(output_lstm_train.shape) + "output_lstm_train shape :" + str(output_lstm_train.shape))

train_generator = MysterySequencer(input_data_train, output_classification_train, NUMBER_OF_CLASSES, BATCH_SIZE)
val_generator = MysterySequencer(input_data_validation, output_classification_validation, NUMBER_OF_CLASSES, BATCH_SIZE)

# Learning
epochs = 50

model = Sequential()
model.add(Dense(1020, activation='softmax', input_shape=(1020,)))


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

tb_callback = TensorBoard(log_dir + experiment_name)
model_checkpoint = ModelCheckpoint(model_dir + experiment_name)
callback_list = [tb_callback, model_checkpoint]

model.fit_generator(
    train_generator,
    steps_per_epoch=INPUT_SIZE // BATCH_SIZE,
    epochs=epochs,
    callbacks=callback_list,
    validation_data=val_generator)