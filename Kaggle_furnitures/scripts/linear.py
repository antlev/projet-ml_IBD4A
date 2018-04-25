from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense
from keras.callbacks import *
from keras.losses import *
from keras.optimizers import *
from keras.activations import *
import time
import datetime

# Adapt to computer
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
log_dir = "/media/antoine/Linux-1/git/projet-ml_IBD4A/Kaggle_furnitures/log_furnitures/"
batch_size = 1400

def own_callback():
    print("yeah !")
    model.save_weights('first_try.h5')

# Experiment name
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
experiment_name = "test_" + st
# experiment_name = "resized-128_linear_" + st
print(experiment_name)

# Dataset
nb_train_samples = 191129
nb_validation_samples = 6289
img_width, img_height = 128, 128
input_shape = (img_width,img_height,3)

# Learning
epochs = 50

model = Sequential()
model.add((Flatten(input_shape=input_shape)))
model.add(Dense(128, activation='tanh'))
model.add(Activation('softmax'))

# model.compile(loss=sgd(lr=0.1),
#               optimizer=mse,
#               metrics=['accuracy'])

model.compile(loss=kullback_leibler_divergence,
              optimizer=rmsprop(),
              metrics=['accuracy'])

tb_callback = TensorBoard(log_dir + experiment_name)

# Generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255)
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

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

batch=next(train_generator)
X=batch[0]
Y=batch[1]
print(X.shape)
print(Y.shape)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[tb_callback],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')