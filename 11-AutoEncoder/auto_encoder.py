import os
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers, optimizers, Model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


tf.random.set_seed(22)
np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2')

# save image
def save_image(imgs, name):
  new_im = Image.new('L', (280, 280))

  index = 0
  for i in range(0, 280, 28):
    for j in range(0, 280, 28):
      im = imgs[index]
      im = Image.fromarray(im, mode='L')
      new_im.paste(im, (i, j))
      index += 1
  new_im.save(name)

h_dim = 20
batchsz = 512
learing_rate = 1e-3

# load data
mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz*5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


class AE(keras.Model):
  def __init__(self):
    super(AE, self).__init__()

    # Encoder
    # [784 -> 256 -> 128 -> h_dim]
    self.encoder = Sequential([
      layers.Dense(256, activation=tf.nn.relu),
      layers.Dense(128, activation=tf.nn.relu),
      layers.Dense(h_dim)
    ])

    # Decoder
    # [h_dim -> 128 -> 256 -> 784]
    self.decoder = Sequential([
      layers.Dense(128, activation=tf.nn.relu),
      layers.Dense(256, activation=tf.nn.relu),
      layers.Dense(784),
    ])

  def call(self, inputs, training=None):
    # [b, 784] -> [b, 10]
    x = self.encoder(inputs)
    # [b, 10]  -> [b, 784]
    x_hat = self.decoder(x)
    return x_hat

  def build(self, input_shape):
      """构建模型各层"""
      self.encoder.build(input_shape)
      encoder_shape = self.encoder.compute_output_shape(input_shape)
      self.decoder.build(encoder_shape)
      decode_shape = self.decoder.compute_output_shape(encoder_shape)
      self.built = True
      super(AE, self).build(input_shape)

model = AE()
model.build(input_shape=(None, 784))
model.summary()


opt = optimizers.Adam(learning_rate=1e-3)

for epoch in range(100):

  for step, x in enumerate(train_db):

    # [ b, 28, 28] = [b, 784]
    x = tf.reshape(x, [-1, 784])

    with tf.GradientTape() as tape:
      x_rec_logits = model(x)

      rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
      rec_loss = tf.reduce_mean(rec_loss)

    grads = tape.gradient(rec_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    if step % 100 == 0:
      print(epoch, step, float(rec_loss))


    # evaluation
    x = next(iter(test_db))
    logits = model(tf.reshape(x, [-1, 784]))
    x_hat = tf.sigmoid(logits)
    # [b, 784] -> [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])

    # [b, 28, 28] -> [2b, 28, 28]
    x_concat = tf.concat([x, x_hat], axis=0)
    x_concat = x_concat.numpy() * 255
    x_concat = x_concat.astype(np.uint8)
    save_image(x_concat, '11-AutoEncoder/ae_imgs/rec_epoch_%d.png'%epoch)

