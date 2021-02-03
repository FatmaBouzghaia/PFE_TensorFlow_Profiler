import os
import sys

# To block info messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
# Now design your model and train it
BATCH_SIZE=20
def download_dataset():
	#There are 50000 samples in the training dataset. You can select a subset of the traing set hre:
	NB_SAMPLES=50000


	conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

	# prints the Resnet architecture
	conv_base.summary()


	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	x_train = x_train / 255.0
	x_test = x_test / 255.0

	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	print(x_train.shape)
	print(x_test.shape)

	print("There are "+str(x_train.shape[0])+" training samples")
	print("There are "+str(x_test.shape[0])+" testing samples")

	df = pd.DataFrame(list(zip(x_train, y_train)), columns =['x_train', 'y_train']) 

	val = df.sample(NB_SAMPLES) # take only 5% of the samples

	x_train = np.array([ i for i in list(val['x_train'])])
	y_train = np.array([ i for i in list(val['y_train'])])

	print("There are "+str(x_train.shape[0])+" training samples")

	print("There are "+str(x_test.shape[0])+" testing samples")
	return (x_train,x_test,y_train,y_test)

# create the CNN model
def create_cnn_model():
	model = models.Sequential()
	model.add(layers.UpSampling2D((2,2)))
	model.add(layers.UpSampling2D((2,2)))
	model.add(layers.UpSampling2D((2,2)))
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(10, activation='softmax'))
	return model

# Train the model and evaluating it
def train_model(model, batch_size):
	x_train,x_test,y_train,y_test=download_dataset()	
	model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

	# Create a TensorBoard callback
	logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

	tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

	history = model.fit(x_train, y_train, epochs=5, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),callbacks = [tboard_callback])
	print("Validation:")
	model.evaluate(x_test, y_test)


	history_dict = history.history
	loss_values = history_dict['loss']
	val_loss_values = history_dict['val_loss']

	epochs = range(1, len(loss_values) + 1)

	plt.figure(figsize=(10,10))

	plt.subplot(1,2,1)
	plt.plot(epochs, loss_values, 'bo', label='Training Loss')
	plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig("resnet50_loss.png")

	acc = history_dict['acc']
	val_acc = history_dict['val_acc']

	epochs = range(1, len(loss_values) + 1)

	plt.figure(figsize=(10, 10))
	plt.subplot(1,2,2)
	plt.plot(epochs, acc, 'bo', label='Training Accuracy', c='orange')
	plt.plot(epochs, val_acc, 'b', label='Validation Accuracy', c='orange')
	plt.title('Training and Validation Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig("resnet50_accuracy.png")

def create_arg_parser():
  batch_size_choices = [8,16,32,64,128,256,512,1024,2048]
  parser = argparse.ArgumentParser(description='Configuring the Resnet50 model.')
  parser.add_argument('--batch_size', type=int, choices=batch_size_choices, 
                    default=32, required=True,
                    help='Setting the batch size for the training and evalution of the model')
  parser.add_argument('--gpu_mode', type=int, choices=[0,1], 
                    default=0, required=True,
                    help='Mode of the GPU: private or not')
  parser.add_argument('--mixed_precision', type=int, choices=[0,1], 
                    default=0, required=True,
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
