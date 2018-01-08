import keras
from keras.metrics import categorical_accuracy
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from keras.losses import *
from keras.datasets import *
from keras.callbacks import *


import numpy as np

experiment_name = "C2D-64-(3-3)_C2D-64-(3-3)_MP2D-(2-2)_DP-0.5-64T_DP-0.5_64T_DP-0.5_64T_DP-0.5_64T_10S_0.5"

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

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('tanh'))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('tanh'))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('tanh'))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('tanh'))
model.add(Dense(10, activation=sigmoid))

# Fit
model.compile(sgd(lr=0.5), mse, metrics=[categorical_accuracy])

model.fit(x_train, y_train,
                 batch_size=1000,
                 epochs=10000,
                 callbacks=[tb_callback],
                 validation_data=(x_test, y_test))

# Save
model.save('./models/' + experiment_name)