from dataset import *
from model import * 

train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg', shuffle=cfg.BUFFER_SIZE)
train_dataset = train_dataset.map(lambda x:load_image(x, True))
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
test_dataset = test_dataset.map(lambda x:load_image(x, False))
test_dataset = test_dataset.batch(1)