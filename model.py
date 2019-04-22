import tensorflow as tf
import config as cfg

class DownSampleGen(tf.keras.Model):

	def __init__(self, filters, size, apply_batchnorm=True):
		super(DownSampleGen, self).__init__()
		self.apply_batchnorm = apply_batchnorm

		initializer = tf.random_normal_initializer(0., 0.02)

		self.conv1 = tf.keras.layers.Conv2D(filters, (size, size), strides=2, padding='SAME', kernel_initializer=initializer, use_bias=False)

		if self.apply_batchnorm:
			self.batchnorm = tf.keras.layers.BatchNormalization()

	def call(self, x, training):
		x = self.conv1(x)
		if self.apply_batchnorm:
			x = self.batchnorm(x, training=training)
		x = tf.nn.leaky_relu(x)

		return x

class UpSample(tf.keras.Model):

	def __init__(self, filters, size, apply_dropout=False):
		super(UpSample, self).__init__()
		self.apply_dropout = apply_dropout

		initializer = tf.random_normal_initializer(0., 0.02)

		self.up_conv = tf.keras.layers.Conv2DTranspose(filters, (size, size), strides=2, padding='SAME', kernel_initializer=initializer, use_bias=False)
		self.batchnorm = tf.keras.layers.BatchNormalization()

		if self.apply_dropout:
			self.dropout = tf.keras.layers.Dropout(0.5)

	def call(self, x1, x2, training):
		x = self.up_conv(x1)
		x = self.batchnorm(x, training=training)
		if self.apply_dropout:
			x = self.dropout(x, training=training)
		x = tf.nn.relu(x)
		x = tf.concat([x, x2], axis=-1)

		return x

class Generator(tf.keras.Model):

	def __init__(self):
		super(Generator, self).__init__()
		initializer = tf.random_normal_initializer(0., 0.02)

		self.down1 = DownSampleGen(64, 4, apply_batchnorm=False)
		self.down2 = DownSampleGen(128, 4)
		self.down3 = DownSampleGen(256, 4)
		self.down4 = DownSampleGen(512, 4)
		self.down5 = DownSampleGen(512, 4)
		self.down6 = DownSampleGen(512, 4)
		self.down7 = DownSampleGen(512, 4)
		self.down8 = DownSampleGen(512, 4)

		self.up1 = UpSample(512, 4, apply_dropout=True)
		self.up2 = UpSample(512, 4, apply_dropout=True)
		self.up3 = UpSample(512, 4, apply_dropout=True)
		self.up4 = UpSample(512, 4)
		self.up5 = UpSample(256, 4)
		self.up6 = UpSample(64, 4)

		self.last = tf.keras.layers.Conv2DTranspose(cfg.OUTPUT_CHANNELS, (4, 4), strides=2, padding='SAME', kernel_initializer=initializer)

	@tf.contrib.eager.defun
	def call(self, x, training):
		x1 = self.down1(x, training=training)
		x2 = self.down2(x1, training=training)
		x3 = self.down3(x2, training=training)
		x4 = self.down4(x3, training=training)
		x5 = self.down5(x4, training=training)
		x6 = self.down6(x5, training=training)
		x7 = self.down2(x6, training=training)
		x8 = self.down2(x7, training=training)

		x9 = self.up1(x8, x7, training=training)
		x10 = self.up2(x9, x6, training=training)
		x11 = self.up3(x10, x5, training=training)
		x12 = self.up4(x11, x4, training=training)
		x13 = self.up5(x12, x3, training=training)
		x14 = self.up6(x13, x2, training=training)
		x15 = self.up1(x14, x1, training=training)

		x16 = self.last(x15)
		x16 = tf.nn.tanh(x16)

		return x16

class DownSampleDis(tf.keras.Model):

	def __init__(self, filters, size, apply_batchnorm=True):
		super(DownSampleDis, self).__init__()
		self.apply_batchnorm = apply_batchnorm

		initializer = tf.random_normal_initializer(0., 0.02)

		self.conv1 = tf.keras.layers.Conv2D(filters, (size, size), strides=2, padding='SAME', kernel_initializer=initializer, use_bias=False)

		if self.apply_batchnorm:
			self.batchnorm = tf.keras.layers.BatchNormalization()

	def call(self, x, training):
		x = self.conv1(x)
		if self.apply_batchnorm:
			x = self.batchnorm(x, training=training)

		x = tf.nn.leaky_relu(x)

		return x

class Discriminator(tf.keras.Model):