from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def check_missing_img(data_dir, test):
    missing = []
    if test:
        nb_img = nb_test_img
    else:
        nb_img = nb_val_img
    for i in range(1, nb_img):
        if os.path.isfile(data_dir + "1/" + str(i) + ".jpg") == False:
            missing.append(str(i))
    return missing


model_path = "../modelscheck_R128_P1_cc-sgd_C2D_16(3-3)mp(2-2)_128sm_2018-05-25_13:32:52"
validation_data_dir = "../data/validation/"
test_data_dir = "../data/test/"
val_submission_file = "../submission/val.sub"
test_submission_file = "../submission/test.sub"

img_width=128
img_height=128
img_resized = (img_width,img_height)
nb_test_img = 12800
nb_val_img = 6289
batch_size = 10
nb_batch = 128
nb_batch_val = 628
nb_class=128

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model = load_model(model_path)

evaluation = model.evaluate_generator(validation_generator, nb_batch_val)

print(evaluation)

results = model.predict_generator(validation_generator, nb_batch_val)
results_to_file = open(val_submission_file, "w")

val_missing = check_missing_img(validation_data_dir, False)
print(str(len(val_missing)) + " validation images are missing")

k=0
for i in range(1, nb_val_img):
    if str(i) in val_missing:
        results_to_file.write(str(i) + "," + str(random.randint(1,128)) + "\n")
    else:
        prob=0
        class_nb=0
        for j in range(1, nb_class):
            if results[k][j] > prob:
                class_nb=j
                prob=results[k][j]
        results_to_file.write(str(i) + "," + str(class_nb) + "\n")
        k=k+1

results_to_file.close()

# Prediction to Kaggle that returns us a poor result
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

results = model.predict_generator(test_generator, nb_batch)
results_to_file = open(test_submission_file, "w")

test_missing = check_missing_img(test_data_dir, True)
print(str(len(test_missing)) + " test images are missing")

k=0
for i in range(1, nb_test_img):
    if str(i) in test_missing:
        results_to_file.write(str(i) + "," + str(random.randint(1,128)) + "\n")
    else:
        prob=0
        class_nb=0
        for j in range(1, nb_class):
            if results[k][j] > prob:
                class_nb=j
                prob=results[k][j]
        results_to_file.write(str(i) + "," + str(class_nb) + "\n")
        k=k+1

results_to_file.close()

