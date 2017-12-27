from keras.metrics import categorical_accuracy
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from keras.losses import *
from keras.datasets import *
from keras.callbacks import *
import keras

import numpy as np

experiment_name = "CONV2D-16-3-3_MAXPOOL2D-2-2_-32-32_0-5"

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
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(32, activation=tanh))
model.add(Dense(32, activation=tanh))
model.add(Dense(10, activation=sigmoid))

# Fit
model.compile(sgd(lr=0.5), mse, metrics=[categorical_accuracy])

model.fit(x_train, y_train,
                 batch_size=4096,
                 epochs=100000,
                 callbacks=[tb_callback],
                 validation_data=(x_test, y_test))

# Save
model.save('./models/' + experiment_name)