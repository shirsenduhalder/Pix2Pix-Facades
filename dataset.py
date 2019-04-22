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

	input_image = tf.image.resize_images(input_image, size=[cfg.IMG_HEIGHT, cfg.IMG_WIDTH], align_corners=True, method=2)
	real_image = tf.image.resize_images(real_image, size=[cfg.IMG_HEIGHT, cfg.IMG_WIDTH], align_corners=True, method=2)

	return input_image, real_image
