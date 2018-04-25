from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model_path = "../models/test"
test_data_dir = "../data/test/"
submission_file = "../submission/test.sub"

img_width = 128
img_height = 128

batch_size = 1400

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


model = load_model(model_path)

prediction = model.predict_generator(test_generator)

prediction_csv = prediction.to_csv(submission_file)

