# from keras.metrics import categorical_accuracy
# from keras.models import *
# from keras.layers import *
# from keras.activations import *
# from keras.optimizers import *
# from keras.losses import *
# from keras.datasets import *
# from keras.callbacks import *
# import keras
#
# import numpy as np
#
# # experiment_name = "10LR0.1-10S_0.5"
# experiment_name = "TEST"
# tb_callback = keras.callbacks.TensorBoard("./logs/"+experiment_name)
#
# (x_train, y_train), (x_test,y_test) = cifar10.load_data() # Load cifar10 data from internet
#
# print(x_train.shape)
# print(y_train.shape)
#
# print(x_train[0:1])
#
# # Normalisation
# x_train = np.reshape(x_train, (-1,32*32*3)) / 255.0
# x_test = np.reshape(x_test, (-1,32*32*3)) / 255.0
# # x_train = x_train  / 255.0
# # x_test = x_test  / 255.0
#
# y_train = keras.utils.to_categorical(y_train)
# y_test = keras.utils.to_categorical(y_test)
#
# print("After reshape")
# print(x_train.shape)
#
#
# # Configuration
# model = Sequential()
# model.add(keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000))
# # model.add(Dense(10, activation=LeakyReLU(0.1), input_dim=32*32*3))
# # model.add(Dense(10, activation=tanh, input_dim=32*32*3))
# # model.add(Dense(10, activation=sigmoid))
#
# # model = Sequential()
# # model.add(Conv2D(16, (2, 2), padding='same', input_shape=(32, 32, 3)))
# # model.add(MaxPool3D((2,2,2)))
# # model.add(Flatten())
# # model.add(Conv2D(64, (3, 3), padding='same'))
# # model.add(MaxPool2D((2,2)))
# # model.add(Flatten())
# #
# # model.add(Dense(64, activation=tanh))
# # model.add(Dense(64, activation=tanh))
# # model.add(Dense(64, activation=tanh))
# # model.add(Dense(64, activation=tanh))
# # model.add(Dense(10, activation=sigmoid, input_dim=32*32*3 ))
#
# # Fit
# model.compile(sgd(lr=0.1), mse, metrics=[categorical_accuracy])
#
# model.fit(x_train, y_train,
#                  batch_size=12288,
#                  epochs=100000,
#                  callbacks=[tb_callback],
#                  validation_data=(x_test, y_test))

import keras
from keras.metrics import categorical_accuracy
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from keras.losses import *
from keras.datasets import *
from keras.callbacks import *
import resnet
import numpy as np

experiment_name = "resnet_test"

# Configure tensorboard
tb_callback = keras.callbacks.TensorBoard("./logs/"+experiment_name)

# Loading data
(x_train, y_train), (x_test,y_test) = cifar10.load_data() # Load cifar10 data from internet

print("Before reshape :")
print(x_train.shape)
print(y_train.shape)

# Normalisation
# x_train = np.reshape(x_train, (-1,32*32*3)) / 255.0
# x_test = np.reshape(x_test, (-1,32*32*3)) / 255.0
x_train = x_train  / 255.0
x_test = x_test  / 255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print("After reshape :")
print(x_train.shape)
print(y_train.shape)


# Configuration
# model = Sequential()
# model.add(Dense(10, activation=tanh, input_dim=32*32*3))
# model.add(Dense(10, activation=sigmoid))

img_rows, img_cols = 32, 32
img_channels = 3
nb_classes = 10

#selon l'utilisation de l'optimizer baisser le batch_size drastiquement (style 200) et l'epoch aussi (500)
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
# Fit
model.compile(sgd(lr=0.5), mse, metrics=[categorical_accuracy])
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[categorical_accuracy])
#ci dessus c'est l'optimizer

model.fit(x_train, y_train,
               batch_size=1000,
               epochs=10000,
               callbacks=[tb_callback],
               validation_data=(x_test, y_test))

# Save
model.save('./models/' + experiment_name)
