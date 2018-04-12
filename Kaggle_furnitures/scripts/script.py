from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import *
from keras.losses import *
from keras.optimizers import *

import itertools
import time
import datetime

# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 191129
nb_validation_samples = 6289
epochs = 50
batch_size = 1400

input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss=kullback_leibler_divergence,
              optimizer=rmsprop(),
              metrics=['accuracy'])

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
experiment_name = "test_" + st
print(experiment_name)
tb_callback = TensorBoard("/media/antoine/Linux-1/esgi/machine-learning/kaggle_furnitures/log_furnitures/" + experiment_name)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
data_format="channels_last")
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
vaidation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = vaidation_datagen.flow_from_directory(
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