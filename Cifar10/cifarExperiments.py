from keras.datasets import cifar10
from keras.utils import *
from keras.layers import *
from keras.callbacks import TensorBoard
from keras.models import *
from keras.metrics import *
from keras.activations import *
from keras.optimizers import *
import numpy as np
import keras

experiment_name = "10-0.1"
tensor_board = TensorBoard("./logs/" + experiment_name)

#variables variables
learning_rate = 0.01
epochs = 10000

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = np.reshape(x_train, (-1, 32*32*3)) / 255.0
x_test = np.reshape(x_test, (-1, 32*32*3)) / 255.0

model = Sequential()
model.add(Dense(10, activation=sigmoid, input_dim=32*32*3))

model.compile(sgd(lr=learning_rate), mse, metrics=[categorical_accuracy])

model.fit(x_train,
         y_train,
         batch_size=4096,
         epochs=epochs,
         callbacks=[tensor_board],
         validation_data=(x_test, y_test))