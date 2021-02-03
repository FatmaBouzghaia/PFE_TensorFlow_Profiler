# inspired by Tensorflow2 tutorial on CNN:
# https://www.tensorflow.org/tutorials/images/cnn
import os
import sys

# Eliminate the log infos and warnings of TensorFlow.
# Letting only error logs showing.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse
from datetime import datetime
import time

# Number of samples used to train the model
NB_SAMPLES = 50000

# Download the dataset and plotting it
def download_dataset():
  #There are 50000 samples in the training dataset. You can select a subset of the traing set here:
  
  # CIFAR10
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, test_images / 255.0

  print("There are " + str(train_images.shape[0]) + " training samples")

  df = pd.DataFrame(list(zip(train_images, train_labels)), columns =['train_images', 'train_labels']) 

  val = df.sample(NB_SAMPLES) # take only 5% of the samples

  train_images = np.array([ i for i in list(val['train_images'])])
  train_labels = np.array([ i for i in list(val['train_labels'])])

  print("There are " + str(train_images.shape[0]) + " training samples")

  plot_dataset(train_images, train_labels)

  return (train_images, train_labels, test_images, test_labels)

# Plotting the dataset
def plot_dataset(train_images, train_labels):
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

  # plot the dataset
  plt.figure(num=1, figsize=(10,10))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(train_images[i], cmap=plt.cm.binary)
      # The CIFAR labels happen to be arrays, 
      # which is why you need the extra index
      plt.xlabel(class_names[train_labels[i][0]])
  #plt.show()
  plt.savefig("dataset.png")

# create the CNN model
def create_cnn_model():
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

  return model

# Train the model  and evaluating it
def train_model(model, batch_size):
  # Load the dataset
  load_type = "download"
  train_images, train_labels, test_images, test_labels = download_dataset()

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  # Create a TensorBoard callback
  logs = "logs/" + load_type + "_" + str(batch_size) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
  
  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                  histogram_freq = 1,
                                                  profile_batch='500,520')

  history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=10, 
                      validation_data=(test_images, test_labels), callbacks = [tboard_callback])

  # Evaluate the model
  plt.figure(num=2, figsize=(10,10))
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')
  fig_name = "accuracy_" + load_type + ".png" 
  plt.savefig(fig_name)

  test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)
  print(test_acc)

  print("Validation:")
  model.evaluate(test_images, test_labels)

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

def main(argv):
  tf.get_logger().setLevel('ERROR')
  # Create argument parser
  parser = create_arg_parser()
  flags = parser.parse_args(args=argv[1:])
  
  # Setting the batch size
  batch_size = flags.batch_size
  print('Setting batch size = ', batch_size)

  if flags.gpu_mode == 1:
    print('Setting the GPU on private mode')
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

  if flags.mixed_precision == 1:
    print("Setting the mixed precision policy")
    if flags.policy_type == 16:
      # Defining the precision strategy
      policy = mixed_precision.Policy('mixed_float16')
      mixed_precision.set_policy(policy)
    elif flags.policy_type == 32:
      # Defining the precision strategy
      policy = mixed_precision.Policy('float32')
      mixed_precision.set_policy(policy)

    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

  model = create_cnn_model()
  train_model(model, batch_size)


if __name__ == '__main__':
  start_time = time.time()
  main(argv=sys.argv)
  print("Working time: %s seconds" % (time.time() - start_time))
