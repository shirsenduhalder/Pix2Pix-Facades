import os
import tensorflow as tf
tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy as np
import matplotlib.pyplot as plt
import config as cfg

DATA_PATH = 'data'

PATH = os.path.join(DATA_PATH, 'facades/')

def load_image(image_file, is_train):
	image = tf.read_file(image_file)
	image = tf.image.decode_jpeg(image)

	#extracting input image segmented image and real facade image 
	w = tf.shape(image)[1]
	w = w//2
	real_image = image[:, :w, :]
	input_image = image[:, w:, :]

	input_image = tf.cast(input_image, tf.float32)
	real_image = tf.cast(real_image, tf.float32)

	if is_train:
		input_image = tf.image.resize_images(input_image, [286, 286], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		real_image = tf.image.resize_images(real_image, [286, 286], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		stacked_image = tf.stack([input_image, real_image], axis=0)
		cropped_image = tf.random_crop(stacked_image, size=[2, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3])
		input_image, real_image = cropped_image[0], cropped_image[1]

		if np.random.random() > 0.5:
			input_image = tf.image.flip_left_right(input_image)
			real_image = tf.image.flip_left_right(real_image)

	else:
		input_image = tf.image.resize_images(input_image, [256, 256], align_corners=True, method=2)
		real_image = tf.image.resize_images(real_image, [256, 256], align_corners=True, method=2)

	input_image = (input_image/127.5) - 1
	real_image = (real_image/127.5) - 1
	
	return input_image, real_image
