# inspired by Tensorflow2 tutorial on CNN:
# https://www.tensorflow.org/tutorials/images/cnn
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

import argparse
from datetime import datetime
import time

start_time = time.time()

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# Load the cifar-10 dataset
(ds_train, ds_test), ds_info = tfds.load(name="cifar10", split=['train', 'test'], as_supervised=True, with_info=True)

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(128)
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

print("There are " + str(ds_info.splits['train'].num_examples) + " training samples")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# print the CNN architecture
model.summary()

# make the CNN more complex
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Create a TensorBoard callback
logs = "logs/prefetch/" + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                  histogram_freq = 1)

history = model.fit(ds_train, epochs=10, batch_size=256, validation_data=ds_test, callbacks = [tboard_callback])

print("Working time: %s seconds" % (time.time() - start_time))
