import csv
import numpy as np
import math
from random import sample, randint, randrange, random
from PIL import Image, ImageOps, ImageDraw

def random_shadow(image):
	"""Apply random shaddow, this is before the conversion to numpy array.
	Apply to original 320x160 image for best results.
	"""
	image = image.convert('RGBA')
	shadow = Image.new('RGBA', image.size)
	draw_shadow = ImageDraw.Draw(shadow)
	
	# pick random opacity for shadow
	shadow_color = (0, 0, 0, randint(30, 250))
	
	# draw random triangle shadow
	shadow_location = [(randrange(-160, 480), randrange(55, 240)) for _ in range(3)]
	draw_shadow.polygon(shadow_location, fill=shadow_color)
	image.paste(shadow, mask=shadow)
	return image.convert('RGB')

def random_flip(image, steering):
	"""Apply random flip and inverse steering angle
	Before image conversion to numpy array
	"""
	if randint(0, 1):
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
		steering = -steering
	return image, steering

def random_rotation(image):
	return image.rotate(randrange(-3, 3))

def random_crop(image, steering, crop_size=(200, 68)):
	""" randomly crop the image and apply a slight
	steering offset to account for any horizontal translation
	returns image and adjusted steering angle
	"""
	# steering to add per pixle of offset from crop
	steer_per_pixle = 0.003

	x_size, y_size = image.size

	# determine maximum crop offsets
	max_horz_shift =  x_size - crop_size[0]
	max_vert_shift =  y_size - crop_size[1]
	
	# horizontial offset can span full image
	horz_shift = randint(0, max_horz_shift)

	# limit vertical crop offset
	vert_shift =  (y_size - crop_size[1]) // 2 + randint(-20, 20)
	
	# crop
	image = image.crop((horz_shift, vert_shift, horz_shift + crop_size[0], vert_shift + crop_size[1]))

	# adjust steering angle based on horizontal shift
	steering += steer_per_pixle * (max_horz_shift/2 - horz_shift)
	return image, steering

def center_crop(image, crop_size=(200, 68)):
	# center crop the image for use in validation and simulation
	x_size, y_size = image.size
	horz_shift =  (x_size - crop_size[0]) // 2
	vert_shift =  (y_size - crop_size[1]) // 2
	image = image.crop((horz_shift, vert_shift, horz_shift + crop_size[0], vert_shift + crop_size[1]))
	return image

def normalize_image(image):
	# equalize contrast and color then normalize to a range of +/- 1
    image = ImageOps.autocontrast(image, cutoff=5)
    image = ImageOps.equalize(image)
    image = np.asarray(image)
    return image/128 - 1.0

def load_driving_log(data_folder, training=True):
	# load driving data from folder
	key = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
	data = []

	# steering offset value for left and right camera images
	steering_offset = 0.3

	# read in the data log
	with open(data_folder+'driving_log.csv', mode='r') as f:
		reader = csv.DictReader(f, key)
		for row in reader:
			steering_angle = float(row['steering'].strip())
			# only add data from rows with some speed
			if float(row['speed'].strip()) > 10.0:
				if training:
					# check if no steering angle and only allow a portion into the training data set
					if abs(steering_angle) > 0.001 or randrange(100) <= 5:
						# Left Image, add steering offset
						data.append([row['left'].strip(), steering_angle + steering_offset])
						
						# Right Image, subtract steering offset
						data.append([row['right'].strip(), steering_angle - steering_offset])
				else:
					# Center Image
					data.append([row['center'].strip(), steering_angle])	
	print("data set contains %i images" % len(data))
	return data

def batch_generator(data_folder, batch_size, image_size, training):
	""" Generage batches of data from data folder
	This version just uses random samples of the full dataset
	This is much eaiser to implement than trying to shffle and step
	through the whole dataset without possible repetition.
	data_folder -- folder where driving_data.csv and IMG folder reside
	batch_size -- size of batch to generate
	training -- [bool] turn on and off data augmentation
	"""
	data = load_driving_log(data_folder, training)
	img_h, img_w, img_ch = image_size
	crop_size = (img_w, img_h)
	while 1:
		# Select a random sample from the data log for the batch
		random_sample = sample(data, batch_size)
		X_batch = np.empty((batch_size, img_h, img_w, img_ch))
		y_batch = np.empty((batch_size, 1))

		# Process the random batch
		for index, row in enumerate(random_sample):
			steering_angle = row[1]
			image = Image.open(row[0])
			
			if training:
				# apply a random number of shadows
				for _ in range(randint(0, 4)):
					image = random_shadow(image)
				image, steering_angle = random_flip(image, steering_angle)
				image, steering_angle = random_crop(image, steering_angle, crop_size)
			else:
				image = center_crop(image, crop_size)

			X_batch[index] = normalize_image(image)
			y_batch[index] = np.asarray(steering_angle)
		yield X_batch, y_batch