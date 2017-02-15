import numpy as np
import math
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.layers import Input, merge
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

from data_processing import batch_generator

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('train_folder', 'data/train2/', "Folder location for training data")
flags.DEFINE_string('validation_folder', 'data/validation2/', "Folder location for validation data")
flags.DEFINE_integer('epochs', 3, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

# NVIDIA's DAVE 2 model as a benchmark
def dave_2(image_size):
	filters_5x5 = [24, 36, 48]
	filters_3x3 = [64, 64]
	filter_FC = [100, 50, 10]
	img_h, img_w, img_ch = image_size
	input_img = Input(shape=image_size)
	x = input_img

	for filters in filters_5x5:
		x = Convolution2D(filters, 5, 5, border_mode='same', subsample=(2, 2))(x)
		x = Activation('relu')(x)
	
	for filters in filters_3x3:
		x = Convolution2D(filters, 3, 3, border_mode='same')(x)
		x = Activation('relu')(x)

	x = Flatten()(x)

	for filters in filter_FC:
		x = Dense(filters)(x)
		x = Activation('relu')(x)

	x = Dense(1)(x)
	prediction = Activation('linear')(x)

	model = Model(input=input_img, output=prediction)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	return model

# Alternate model
def js4_model(image_size):
	filters_5x5 = [24, 36, 48]
	filters_3x3 = [64, 64]
	filter_FC = [100, 50, 10]
	
	img_h, img_w, img_ch = image_size
	input_img = Input(shape=image_size)
	x = input_img

	# 5x5 convolution filters with dimensionality reduction.
	for filters in filters_5x5:
		res = Convolution2D(filters, 1, 1, border_mode='same', subsample=(2, 2))(x)
		x = Convolution2D(filters, 5, 5, border_mode='same', subsample=(2, 2))(x)
		x = merge([res, x], mode='sum')
		x = Activation('relu')(x)
	
	# 3x3 convolution filters
	for filters in filters_3x3:
		res = Convolution2D(filters, 1, 1, border_mode='same')(x)
		x = Convolution2D(filters//4, 1, 1, border_mode='same', activation='relu')(x)
		x = Convolution2D(filters//4, 3, 3, border_mode='same', activation='relu')(x)
		x = Convolution2D(filters, 1, 1, border_mode='same')(x)
		x = merge([res, x], mode='sum')
		x = Activation('relu')(x)

	x = Flatten()(x)

	# Fully connected layers with dropout
	for filters in filter_FC:
		x = Dense(filters)(x)
		x = Activation('relu')(x)

	x = Dense(1)(x)
	prediction = Activation('tanh')(x)

	model = Model(input=input_img, output=prediction)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	return model

def main(_):
	print("Training:", FLAGS.train_folder)
	print("Validation:", FLAGS.validation_folder)
	print("Epochs:", FLAGS.epochs)
	print("Batch Size:", FLAGS.batch_size)

	# image size after processing, original image is (160, 320, 3)
	image_size = (68, 200, 3)

	# model selction
	# model = dave_2(image_size)
	model = js4_model(image_size)

	print("Model constructed")
	data_set_size = 20000
	validation_set_size = 4000

	# construct batch generators
	train_generator = batch_generator(FLAGS.train_folder, FLAGS.batch_size, image_size, training=True)
	validation_generator = batch_generator(FLAGS.validation_folder, FLAGS.batch_size, image_size, training=False)

	# Take a look at the data set sizes
	print("Training ", end='')
	next(train_generator)
	print("Vallidation ", end='')
	next(validation_generator)

	# Make a nice sample sizes that are divisable by the batch size
	data_per_epoch = (data_set_size//FLAGS.batch_size) * FLAGS.batch_size
	validation_samples = (validation_set_size//FLAGS.batch_size) * FLAGS.batch_size

	model.fit_generator(
		generator=train_generator,
		samples_per_epoch=data_per_epoch,
		nb_epoch=FLAGS.epochs, 
		validation_data=validation_generator,
		nb_val_samples=validation_samples
		)

	# Save model architecture and weights
	model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
	tf.app.run()

""" Calling
python model.py --train_folder data/train2/ --validation_folder data/validation2/ --epochs 2 --batch_size 256
"""