from dataset import *
from model import *
from loss import *
import cv2 

train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg', shuffle=cfg.BUFFER_SIZE)
train_dataset = train_dataset.map(lambda x:load_image(x, True))
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
test_dataset = test_dataset.map(lambda x:load_image(x, False))
test_dataset = test_dataset.batch(1)

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)

EPOCHS = cfg.EPOCHS

def save_images(model, test_input, tar, epoch):
	prediction = model(test_input, training=True)
	test_input, target, predicted_output = test_input[0], tar[0], prediction[0]
	img = np.hstack([test_input, target, predicted_output])

	if not os.path.exists('samples/'):
		os.makedirs('samples/')

	cv2.imwrite("epoch-{}.jpg".format(img))


def train(dataset, epochs):

	for epoch in range(epochs):
		start = time.time()

		for input_image, target in dataset:
			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				gen_output = generator(input_image, training=True)
				disc_real_output = discriminator(input_image, target, training=True)
				disc_generated_output = discriminator(input_image, gen_output, training=True)

				gen_loss = generator_loss(disc_generated_output, gen_output, target)
				disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

				generator_gradients = gen_tape.gradient(gen_loss, generator.variables)
				discrimiator_gradients = disc_tape.gradient(disc_loss, discriminator.variables)

				generator_optimizer.apply_gradients(zip(generator_gradients, generator.variables))
				discriminator_optimizer.apply_gradients(zip(discrimiator_gradients, discriminator.variables))

		if(epoch%1 == 0):
			for inp, tar in test_dataset.take(1):
				save_images(generator, inp, tar, epoch)

		if( (epoch + 1)%20 == 0):
			checkpoint.save(file_prefix=checkpoint_prefix)

		print("Time taken for epoch {} is {} seconds".format((epoch + 1), (time.time() -  start)))

train(train_dataset, EPOCHS)