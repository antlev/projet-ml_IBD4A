from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import *
from keras.losses import *
from keras.optimizers import *
from keras.activations import *
from keras.layers import Conv2D, MaxPooling2D
import time
import datetime

# Adapt to computer
train_data_dir = '/media/antoine/Linux-1/git/projet-ml_IBD4A/Kaggle_furnitures/data/train'
validation_data_dir = '/media/antoine/Linux-1/git/projet-ml_IBD4A/Kaggle_furnitures/data/validation'
log_dir = "/media/antoine/Linux-1/git/projet-ml_IBD4A/Kaggle_furnitures/log_furnitures/"
model_dir = "/media/antoine/Linux-1/git/projet-ml_IBD4A/Kaggle_furnitures/models"
batch_size = 1400

# Experiment name
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
experiment_name = "resized-128_experiment_1" + st
print(experiment_name)

# Dataset
nb_train_samples = 191129
nb_validation_samples = 6289
img_width, img_height = 128, 128
input_shape = (img_width,img_height,3)

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

# Learning
nb_test = 4
epochs = 50

nb_launch = 4

activations_shortcut = {'sigmoid' : 'Si', 'softmax' : 'Sm', 'tanh' : 'T', 'relu': 'R'}

neurons_conf = (
    (64, 64, 128),
    (64, 128, 128),
    (128, 128, 128),
    (256, 128, 128),
    (256, 256, 128),
    (512, 256, 128, 128),
    (512, 512, 512, 512),
    (64, 64, 128),
    (64, 128, 128),
    (128, 128, 128),
    (256, 128, 128),
    (256, 256, 128),
    (512, 256, 128, 128),
    (512, 512, 512, 512),
)
activations_conf = (
    ('tanh', 'tanh', 'softmax'),
    ('tanh', 'tanh', 'softmax'),
    ('tanh', 'tanh', 'softmax'),
    ('tanh', 'tanh', 'softmax'),
    ('tanh', 'tanh', 'softmax'),
    ('tanh', 'tanh', 'tanh', 'softmax'),
    ('tanh', 'tanh', 'tanh', 'softmax'),
    ('relu', 'relu', 'softmax'),
    ('relu', 'relu', 'softmax'),
    ('relu', 'relu', 'softmax'),
    ('relu', 'relu', 'softmax'),
    ('relu', 'relu', 'softmax'),
    ('relu', 'relu', 'relu', 'softmax'),
    ('relu', 'relu', 'relu', 'softmax')
)

# Model construction
for lauch_nb in range(nb_launch):
    for model_nb in range(len(neurons_conf)):
        model = Sequential()
        model_desc = "D"

        for layer_nb in range(len(neurons_conf[model_nb])):
            model.add(Dense(neurons_conf[model_nb][layer_nb], activation=activations_conf[model_nb][layer_nb]))
            model_desc += str(neurons_conf[model_nb][layer_nb]) + activations_shortcut[activations_conf[model_nb][layer_nb]]
            if(layer_nb != len(neurons_conf[model_nb])-1):
                model_desc += '-'

        model.compile(loss=kullback_leibler_divergence,
                      optimizer=rmsprop(),
                      metrics=['accuracy'])
        model_desc += "." + str(lauch_nb)
        tb_callback = TensorBoard(log_dir + experiment_name + '.' + model_desc)

        print("Debug >>> Fitting model >" + model_desc + "< log >" + log_dir + experiment_name + '.' + model_desc + "<")

        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            callbacks=[tb_callback],
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        model.save_weights(model_dir + experiment_name + '.' )