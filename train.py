import os
import numpy as np
import cv2
import keras
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

test_data_path = os.path.join("data", "test")
train_data_path = os.path.join("data", "train")

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(96, 96, 3)))
model.add(keras.layers.Dense(4, activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    train_data_path,
    target_size=(96, 96),
    batch_size=1,
    shuffle=False,
    class_mode="binary")
test_generator = datagen.flow_from_directory(
    test_data_path,
    target_size=(96, 96),
    batch_size=1,
    shuffle=False,
    class_mode="binary")
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples,
    epochs=5,
    validation_data=test_generator,
    validation_steps=test_generator.samples)
model_json = model.to_json()
with open("model.json", "w") as model_json_file:
    model_json_file.write(model_json)
model.save_weights("model.h5")
