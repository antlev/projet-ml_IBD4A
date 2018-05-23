from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def check_missing_img():
    missing = []
    for i in range(1, nb_test_img):
        if os.path.isfile(test_data_dir + "1/" + str(i) + ".jpg") == False:
            missing.append(str(i))
    return missing


model_path = "../models/test"
test_data_dir = "../data/test/"
submission_file = "../submission/test.sub"

img_width=128
img_height=128
img_resized = (img_width,img_height)
nb_test_img = 12800
batch_size = 100
nb_batch = 128
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

missing = check_missing_img()
print(str(len(missing)) + " images are missing")

k=0
for i in range(0, batch_size*nb_batch):
    if str(i) in missing:
        results_to_file.write(str(i+1) + "," + str(random.randint(1,128)) + "\n")
    else:
        prob=0
        class_nb=0
        for j in range(1, nb_class):
            if results[k][j] > prob:
                class_nb=j
                prob=results[k][j]
        results_to_file.write(str(i+1) + "," + str(class_nb) + "\n")
        k=k+1

results_to_file.close()

