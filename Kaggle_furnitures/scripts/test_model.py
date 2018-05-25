from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def check_missing_img(data_dir, nb_img):
    missing = []
    for i in range(1, nb_img):
        if os.path.isfile(data_dir + "1/" + str(i) + ".jpg") == False:
            missing.append(str(i))
    return missing


model_path = "../modelscheck_R128_P1_cc-sgd_C2D_16(3-3)mp(2-2)_128sm_2018-05-25_13:32:52"
validation_data_dir = "../data/validation/"
raw_validation_data_dir = "../data/validationRaw/"
test_data_dir = "../data/test/"
val_submission_file = "../submission/val.sub"
test_submission_file = "../submission/test.sub"

img_width=128
img_height=128
img_resized = (img_width,img_height)
nb_test_img = 12800
nb_val_img = 6400
nb_batch_test = 128
nb_batch_val = 64
batch_size = 100
nb_class=128

print("Loading model...")
model = load_model(model_path)

print("Loading validation data...")
validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print("Evaluate validation data")
evaluation = model.evaluate_generator(validation_generator, nb_batch_val)
print("Model evaluation on validation data : " + str(evaluation))

print("Loading raw validation data...")
raw_validation_datagen = ImageDataGenerator(rescale=1. / 255)

raw_validation_generator = raw_validation_datagen.flow_from_directory(
    raw_validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None)

val_missing = check_missing_img(raw_validation_data_dir, nb_val_img)
print(str(len(val_missing)) + " raw validation images are missing")

print("Predict on validation_Raw...")
raw_val_results = model.predict_generator(raw_validation_generator, nb_batch_val)
raw_val_results_to_file = open(val_submission_file, "w")

k=0
for i in range(1, nb_val_img):
    if str(i) in val_missing:
        raw_val_results_to_file.write(str(i) + ",no image\n")
    else:
        prob=0
        class_nb=0
        for j in range(1, nb_class):
            if raw_val_results[k][j] > prob:
                class_nb=j
                prob=raw_val_results[k][j]
        raw_val_results_to_file.write(str(i) + "," + str(class_nb) + "\n")
        k=k+1

raw_val_results_to_file.close()
print("Finnished written raw validation results")

print("Predict on test...")
# Prediction to Kaggle that returns us a poor result
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

test_missing = check_missing_img(test_data_dir, nb_test_img)
print(str(len(test_missing)) + " test images are missing")

test_results = model.predict_generator(test_generator, nb_batch_test)
test_results_to_file = open(test_submission_file, "w")

k=0
for i in range(1, nb_test_img):
    if str(i) in test_missing:
        test_results_to_file.write(str(i) + "," + str(random.randint(1, 128)) + "\n")
    else:
        prob=0
        class_nb=0
        for j in range(1, nb_class):
            if test_results[k][j] > prob:
                class_nb=j
                prob=test_results[k][j]
        test_results_to_file.write(str(i) + "," + str(class_nb) + "\n")
        k=k+1

test_results_to_file.close()
print("Finnished written test results")

