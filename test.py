import os
import numpy as np
import cv2
import keras
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

test_data_path = os.path.join("data", "test")

def load_data_from_image_files(base_data_path):
    X = []
    y = []
    for data_folder in os.listdir(base_data_path):
        data_folder_path = os.path.join(base_data_path, data_folder)
        if os.path.isdir(data_folder_path):
            for filename in os.listdir(data_folder_path):
                if filename.endswith(".jpg"):
                    X.append(cv2.imread(os.path.join(data_folder_path, filename))[...,::-1]) # reverse channels BGR -> RGB
                    if data_folder == "null":
                        y.append([0])
                    else:
                        y.append([1])
    return np.array(X).astype("float32") / 255.0, np.array(y)

with open("model.json", "r") as json_file:
    model = keras.models.model_from_json(json_file.read())
model.load_weights("model.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

X_test, y_test = load_data_from_image_files(test_data_path)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(96, 96),
    batch_size=1,
    shuffle=False,
    class_mode="binary")
_, generator_test_accuracy = model.evaluate_generator(generator=test_generator, steps=test_generator.samples)
_, test_accuracy = model.evaluate(X_test, y_test)
print("evaluate_generator: %.3f, evaluate: %.3f" % (generator_test_accuracy, test_accuracy))
