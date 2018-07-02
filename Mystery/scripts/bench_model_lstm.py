from keras.callbacks import *
import keras
import time
import datetime
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Reshape
from scripts.my_classes import MysterySequencer, activations_shortcut, loss_shortcut, optimizer_shortcut

# Experiment name
ts = time.time()
date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

experiment_name = "TEST"
print("Launching experiment : " + experiment_name)

# Adapt to computer
input_data_path = 'data/2018_04_28_full_train-000000-input.npy'
output_lstm_path = 'data/2018_04_28_full_train-000000-output2.npy'
log_dir = "logs/"
model_dir = "models/"

RANDOM_SEED = 42
INPUT_SIZE = 15444000
NUMBER_OF_CLASSES = 1020
BATCH_SIZE = 150000
INPUT_SHAPE=(4,255)
# Learning
EPOCHS = 3
NB_LAUNCH = 1

layers_conf = (
    ('LSTM', 'D'),
    ('LSTM', 'D'),
    ('LSTM', 'D')
)
layers_dim = (
    (16, 30),
    (32, 30),
    (64, 30)
)
activations_conf = (
    ('softmax', 'softmax'),
    ('softmax', 'softmax'),
    ('softmax', 'softmax')
)
# Compilation
loss_conf = (
    ('categorical_crossentropy'),
    ('categorical_crossentropy'),
    ('categorical_crossentropy')
)
optimizer_conf = (
    ('rmsprop'),
    ('rmsprop'),
    ('rmsprop')
)

# Load data
input_data=np.load(input_data_path, mmap_mode='r')
output_lstm=np.load(output_lstm_path, mmap_mode='r')
input_data = input_data.reshape(-1,4,255)
input_data_train, input_data_validation = np.split(input_data, [int(.7*len(input_data))])
output_lstm_train, output_lstm_validation = np.split(output_lstm, [int(.7*len(output_lstm))])

# Generator
train_generator = MysterySequencer(input_data_train, output_lstm_train, NUMBER_OF_CLASSES, BATCH_SIZE, False)
val_generator = MysterySequencer(input_data_validation, output_lstm_validation, NUMBER_OF_CLASSES, BATCH_SIZE, False)

# Model construction
for lauch_nb in range(NB_LAUNCH):
    for model_nb in range(len(layers_conf)):
        model = keras.models.Sequential()
        model_desc = ""
        model_desc += loss_shortcut[loss_conf[model_nb]] + "-" + optimizer_shortcut[optimizer_conf[model_nb]] + "_"

        for layer_nb in range(len(layers_conf[model_nb])):
            if layer_nb == 0:
                if layers_conf[model_nb][layer_nb] == 'C2D':
                    model.add(Conv2D(layers_dim[model_nb][layer_nb], (2,2), activation=activations_conf[model_nb][layer_nb], input_shape=INPUT_SHAPE))
                    model_desc += str(layers_conf[model_nb][layer_nb]) + '-' + str(layers_dim[model_nb][layer_nb]) + "(2,2)" + \
                                  activations_shortcut[activations_conf[model_nb][layer_nb]]
                if layers_conf[model_nb][layer_nb] == 'D':
                    model.add(Dense(layers_dim[model_nb][layer_nb], activation=activations_conf[model_nb][layer_nb], input_shape=INPUT_SHAPE))
                    model_desc += str(layers_conf[model_nb][layer_nb]) + '-' + str(layers_dim[model_nb][layer_nb]) + "-" + activations_shortcut[activations_conf[model_nb][layer_nb]]
                if layers_conf[model_nb][layer_nb] == 'LSTM':
                    model.add(LSTM(layers_dim[model_nb][layer_nb], activation=activations_conf[model_nb][layer_nb], input_shape=INPUT_SHAPE))
                    model_desc += str(layers_conf[model_nb][layer_nb]) + '-' + str(layers_dim[model_nb][layer_nb]) + "-" + activations_shortcut[activations_conf[model_nb][layer_nb]]
            else:
                if layers_conf[model_nb][layer_nb] == 'C2D':
                    model.add(Conv2D(layers_dim[model_nb][layer_nb], (2,2), activation=activations_conf[model_nb][layer_nb]))
                    model_desc += str(layers_conf[model_nb][layer_nb]) + '-' + str(layers_dim[model_nb][layer_nb]) + "(2,2)" + \
                                  activations_shortcut[activations_conf[model_nb][layer_nb]]
                if layers_conf[model_nb][layer_nb] == 'MP2D':
                    model.add(MaxPooling2D(pool_size=(2,2)))
                    model_desc += str(layers_conf[model_nb][layer_nb]) + "(2,2)"
                if layers_conf[model_nb][layer_nb] == 'D':
                    model.add(Dense(layers_dim[model_nb][layer_nb], activation=activations_conf[model_nb][layer_nb]))
                    model_desc += str(layers_conf[model_nb][layer_nb]) + '-' + str(layers_dim[model_nb][layer_nb]) + "-" + activations_shortcut[activations_conf[model_nb][layer_nb]]
                if layers_conf[model_nb][layer_nb] == 'LSTM':
                    model.add(Reshape((4,255)))
                    model.add(LSTM(layers_dim[model_nb][layer_nb], activation=activations_conf[model_nb][layer_nb]))
                    model_desc += str(layers_conf[model_nb][layer_nb]) + '-' + str(layers_dim[model_nb][layer_nb]) + "-" + activations_shortcut[activations_conf[model_nb][layer_nb]]

            if(layer_nb != len(layers_conf[model_nb])-1):
                model_desc += '-'

        model.compile(loss=loss_conf[model_nb],
                      optimizer=optimizer_conf[model_nb],
                      metrics=['accuracy'])

        model_desc += "." + str(lauch_nb)
        tb_callback = TensorBoard(log_dir + experiment_name + '.' + model_desc)
        model_checkpoint = ModelCheckpoint(model_dir + experiment_name + model_desc)

        callback_list = [tb_callback, model_checkpoint]

        print(">>> Fitting model >" + model_desc + "< log >" + log_dir + experiment_name + model_desc + "<")

        model.fit_generator(
            train_generator,
            steps_per_epoch=INPUT_SIZE // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callback_list,
            validation_data=val_generator)