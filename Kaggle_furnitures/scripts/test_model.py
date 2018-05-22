from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

model_path = "../models/test"
test_data_dir = "../data/test/"
submission_file = "../submission/test.sub"

img_width=128
img_height=128
img_resized = (img_width,img_height)
nb_test_img = 12510
batch_size = 1251
nb_batch = 10
nb_class=128

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

model = load_model(model_path)

results = model.predict_generator(test_generator, nb_batch)

results_to_file = open(submission_file, "w")

for i in range(batch_size*nb_batch):
    prob=0
    class_nb=0
    for j in range(nb_class):
        if results[i][j] > prob:
            class_nb=j
            prob=results[i][j]
    results_to_file.write(str(i) + "," + str(class_nb) + "\n")

results_to_file.close()
