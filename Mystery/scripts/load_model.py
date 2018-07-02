from keras.models import load_model
from keras.callbacks import *
from scripts.my_classes import MysterySequencer

# Adapt to computer
log_dir = "log_furnitures/"
model_dir = "models/"
input_data_path = 'data/2018_04_28_full_train-000000-input.npy'
output_lstm_path = 'data/2018_04_28_full_train-000000-output2.npy'
model_desc="MYST_LSTM_CC_RM0.01_128_2018-06-24_12:08:57"
# Dataset
nb_train_samples = 191129
nb_validation_samples = 6289
img_width, img_height = 128, 128
input_shape = (img_width,img_height,3)
# Learning
INITIAL_EPOCH=10
EPOCHS=20
BATCH_SIZE = 250000
INPUT_SIZE = 15444000
NUMBER_OF_CLASSES = 1020
INPUT_SHAPE=(4,255)

# Load data
input_data=np.load(input_data_path, mmap_mode='r')
output_lstm=np.load(output_lstm_path, mmap_mode='r')
# Before preprocessing
input_data = input_data.reshape(-1,4,255)
# Split train validation
input_data_train, input_data_validation = np.split(input_data, [int(.7*len(input_data))])
output_lstm_train, output_lstm_validation = np.split(output_lstm, [int(.7*len(output_lstm))])

# Generator
train_generator = MysterySequencer(input_data_train, output_lstm_train, NUMBER_OF_CLASSES, BATCH_SIZE, False)
val_generator = MysterySequencer(input_data_validation, output_lstm_validation, NUMBER_OF_CLASSES, BATCH_SIZE, False)

tb_callback = TensorBoard(log_dir + model_desc)
model_checkpoint = ModelCheckpoint(model_dir + model_desc + ".new", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [tb_callback, model_checkpoint]

model = load_model(model_dir + model_desc)

# Fit model
model.fit_generator(
    train_generator,
    steps_per_epoch=INPUT_SIZE // BATCH_SIZE,
    initial_epoch=INITIAL_EPOCH,
    epochs=EPOCHS,
    callbacks=callback_list,
    validation_data=val_generator)