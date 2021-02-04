# inspired by Tensorflow2 tutorial on CNN:
# https://www.tensorflow.org/tutorials/images/cnn
import os
import sys

# To disable info and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_datasets as tfds

import argparse
from datetime import datetime
import time

# Start time of the application
start_time = time.time()

# Create the argument parser
def create_arg_parser():
  batch_size_choices = [8,16,32,64,128,256,512,1024,2048, 4096, 8192,16384,32768]
  parser = argparse.ArgumentParser(description='Configuring the CNN model.')
  parser.add_argument('--batch_size', type=int, choices=batch_size_choices, 
                    default=32, required=True,
                    help='Setting the batch size for the training and evalution of the model')
  parser.add_argument('--gpu_mode', type=int, choices=[0,1], 
                    default=0, required=False,
                    help='Mode of the GPU: private or not')
  parser.add_argument('--mixed_precision', type=int, choices=[0,1], 
                    default=0, required=False,
                    help='Define the mixed precision strategy or not')
  parser.add_argument('--policy_type', type=int, choices=[16, 32],
                    default=16, required=False,
                    help="Setting the policy on: 16 for mixed_float16 or 32 for float32")
  return parser

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# Getting the arguments
parser = create_arg_parser()
argv = sys.argv
flags = parser.parse_args(args=argv[1:])

# Setting the batch size
batch_size = flags.batch_size
print('Setting batch size = ', batch_size)

if flags.gpu_mode == 1:
  print('Setting the GPU on private mode')
  os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

if flags.mixed_precision == 1:
  print("Setting the mixed precision policy")
  # Defining the precision strategy
  if flags.policy_type == 16:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
  elif flags.policy_type == 32:
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_policy(policy)

  print('Compute dtype: %s' % policy.compute_dtype)
  print('Variable dtype: %s' % policy.variable_dtype)

# Load the cifar-10 dataset
(ds_train, ds_test), ds_info = tfds.load(name="cifar10",
                                        split=['train', 'test'],
                                        as_supervised=True,
                                        with_info=True)

ds_train = ds_train.map(normalize_img,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(128)
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
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

# Compile the CNN model
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# Create a TensorBoard callback
logs = "log_prefetch/" + str(batch_size) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                  histogram_freq = 1,
                                                  profile_batch='500,520')
# Train the model
history = model.fit(ds_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=ds_test,
                    callbacks = [tboard_callback])

print("Working time: %s seconds" % (time.time() - start_time))
