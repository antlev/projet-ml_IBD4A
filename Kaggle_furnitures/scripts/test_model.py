from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

model_path = "models/R128_P1_cc-sgd_C2D_16(3-3)mp(2-2)_128sm_2018-05-25_19:22:53"
validation_data_dir = "data/validation/"
test_data_dir = "data/test/"
raw_validation_data_dir = "data/validationRaw/"
val_submission_file = "submission/val.sub"
test_submission_file = "submission/test.sub"

img_width=128
img_height=128
img_resized = (img_width,img_height)
nb_test_img = 12800
nb_val_img = 6400
nb_batch_test = 128
nb_batch_val = 64
batch_size = 100
nb_class=128

def check_missing_img(data_dir, nb_img):
    missing = []
    for i in range(1, nb_img+1):
        if os.path.isfile(data_dir + "1/" + str(i) + ".jpg") == False:
            missing.append(str(i))
    return missing

def check_result(line, data_dir):
    line=line.split(',')
    return os.path.isfile(data_dir + str(line[1]).rstrip() + "/" + str(line[0]) + "_" + str(line[1]).rstrip() + ".jpg")

def check_accuracy(solution_file, data_dir):
    true_res=0
    false_res=0
    with open(solution_file) as fp:
        line = fp.readline()
        while line:
            if check_result(line, data_dir) == False:
                false_res+=1
            else:
                true_res+=1
            line = fp.readline()
    print("True : " +  str(true_res) + " - False : " + str(false_res) + " - Accuracy = " + str(100*true_res/(true_res+false_res))+ "%")

def write_results(results, nb_img, submission_file):
    results_to_file = open(submission_file, "w")
    result_nb=0
    for img_nb in range(1, nb_img+1):
        if str(img_nb) in test_missing:
            results_to_file.write(str(img_nb) + "," + str(random.randint(1, 128)) + "\n")
        else:
            prob=0
            class_predicted=0
            for class_nb in range(nb_class):
                if test_results[result_nb][class_nb] > prob:
                    class_predicted=class_nb
                    prob=test_results[result_nb][class_nb]
            results_to_file.write(str(img_nb) + "," + str(class_predicted) + "\n")
            result_nb=result_nb + 1
    results_to_file.close()

def init_generator(data_dir, is_test):
    datagen = ImageDataGenerator(rescale=1. / 255)
    if is_test == True:
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
        print ("toto")
    else:
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    return generator


print("Loading model...")
model = load_model(model_path)

print("Loading validation data...")
validation_generator = init_generator(validation_data_dir, False)
print("Evaluate validation data...")
evaluation = model.evaluate_generator(validation_generator, nb_batch_val)
print("Model evaluation on validation data : " + str(evaluation))

print("Loading raw validation data...")
raw_validation_generator = init_generator(raw_validation_data_dir, True)
val_missing = check_missing_img(raw_validation_data_dir, nb_val_img)
print(str(len(val_missing)) + " raw validation images are missing")
print("Predict on validation_Raw...")
raw_val_results = model.predict_generator(raw_validation_generator, nb_batch_val)
write_results(raw_val_results, nb_val_img, val_submission_file)
print("Finnished written raw validation results")

print("Predict on test...")
test_generator = init_generator(test_data_dir, True)
test_missing = check_missing_img(test_data_dir, nb_test_img)
print(str(len(test_missing)) + " test images are missing")
test_results = model.predict_generator(test_generator, nb_batch_test)
write_results(test_results, nb_test_img, test_submission_file)
print("Finnished written test results")

check_accuracy(val_submission_file, validation_data_dir)