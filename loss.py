import tensorflow as tf
LAMBDA = 100

def discriminator_loss(disc_real_output, disc_generated_output):

	real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_real_output), logits=disc_real_output)

	generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(disc_generated_output),logits=disc_generated_output)

	total_disc_loss = real_loss + generated_loss

	return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):

	gan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)
	l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
	total_gan_loss = tf.add(gan_loss, tf.multiply(LAMBDA, l1_loss))

	return total_gan_loss