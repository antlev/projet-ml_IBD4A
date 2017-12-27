from keras.metrics import categorical_accuracy
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from keras.losses import *
from keras.datasets import *
import keras

import numpy as np

experiment_name = "MNIST_BASELINE_10_LR_0.5"

# Loading data
(x_train, y_train), (x_test,y_test) = mnist.load_data()

print("Before reshape :")
print(x_train.shape)
print(y_train.shape)

# Normalisation
x_train = np.reshape(x_train, (-1,28*28)) / 255.0
x_test = np.reshape(x_test, (-1,28*28)) / 255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print("After reshape :")
print(x_train.shape)
print(y_train.shape)

# Callback
tb_callback = keras.callbacks.TensorBoard("./logs/"+experiment_name)

# Configuration
model = Sequential()

model.add(Dense(10, activation=tanh, input_dim=784))
model.add(Dense(10, activation=sigmoid))

# Fit
model.compile(sgd(lr=0.5), mse, metrics=[categorical_accuracy])

model.fit(x_train, y_train,
                 batch_size=4096,
                 epochs=1000,
                 callbacks=[tb_callback],
                 validation_data=(x_test, y_test))

model.save('./models/' + experiment_name)