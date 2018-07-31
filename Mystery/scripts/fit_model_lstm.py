from keras.callbacks import *
import keras
import time
import datetime
import numpy as np
from keras.layers import Dense, Activation
from scripts.my_classes import MysterySequencer
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# Experiment name
ts = time.time()
date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
# experiment_name = "MYST_LSTM_CC_RM0.01_256_" + date_time
experiment_name = "MYST_LSTM_CC_ADAM_128_BC5000_" + date_time
print("Launching experiment : " + experiment_name)

# Adapt to computer
input_data_path = 'data/2018_04_28_full_train-000000-input.npy'
output_lstm_path = 'data/2018_04_28_full_train-000000-output2.npy'
log_dir = "logs/"
model_dir = "modelsLstm/"

INPUT_SIZE = 15444000
NUMBER_OF_CLASSES = 1020
BATCH_SIZE = 50000
INPUT_SHAPE=(4,255)
# Learning
EPOCHS = 10

# Load data
input_data=np.load(input_data_path, mmap_mode='r')
output_lstm=np.load(output_lstm_path, mmap_mode='r')

# Before preprocessing
# print("Raw data :")
# print("input_data shape : " + str(input_data.shape) + " - input_data[0] size : " + str(input_data[0].size) + " - different values are : " + str(all_diff_element(input_data[0], True)))
# print("output_data shape : " + str(output_lstm.shape) + " - different values are : " + str(all_diff_element(output_lstm[0], True)))

input_data = input_data.reshape(-1,4,255)

# Split train validation
input_data_train, input_data_validation = np.split(input_data, [int(.7*len(input_data))])
output_lstm_train, output_lstm_validation = np.split(output_lstm, [int(.7*len(output_lstm))])

print("After split :")
print("input_data_train shape :" + str(input_data_train.shape) + " - input_data_validation shape :" + str(input_data_validation.shape) )
print("output_lstm_train shape :" + str(output_lstm_train.shape) + " - output_lstm_train shape :" + str(output_lstm_train.shape))

# Generator
train_generator = MysterySequencer(input_data_train, output_lstm_train, NUMBER_OF_CLASSES, BATCH_SIZE, False)
val_generator = MysterySequencer(input_data_validation, output_lstm_validation, NUMBER_OF_CLASSES, BATCH_SIZE, False)

# Model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, input_shape=INPUT_SHAPE))
model.add(Dense(30))
model.add(Activation('softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Callback
tb_callback = TensorBoard(log_dir + experiment_name)
model_checkpoint = ModelCheckpoint(model_dir + experiment_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [tb_callback, model_checkpoint]

# Fit model
model.fit_generator(
    train_generator,
    steps_per_epoch=INPUT_SIZE // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callback_list,
    validation_data=val_generator)