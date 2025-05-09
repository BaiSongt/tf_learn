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


z_dim = 10

class VAE(keras.Model):
  def __init__(self):
    super(VAE, self).__init__()

    # Encoder
    self.fc1 = layers.Dense(128)
    self.fc2 = layers.Dense(z_dim) # get mean prediction
    self.fc3 = layers.Dense(z_dim)
    # Decoder
    self.fc4 = layers.Dense(128)
    self.fc5 = layers.Dense(784)

  def encoder(self, x):
    h = tf.nn.relu(self.fc1(x))
    mu = self.fc2(h) # get mean
    log_var = self.fc3(h) # get log variance
    return mu, log_var

  def decoder(self, z):
    out = tf.nn.relu(self.fc4(z))
    return self.fc5(out)

  def reparameterize(self, mu, log_var):
    eps = tf.random.normal(log_var.shape)
    std = tf.exp(log_var) ** 0.5
    z = mu + std * eps
    return z


  def call(self, inputs, training=None):
    # [b, 784] -> [b, z_dim], [b, z_dim]
    mu, log_var = self.encoder(inputs)

    # reparameterization trick
    z = self.reparameterize(mu, log_var)

    x_hat = self.decoder(z)

    return x_hat, mu, log_var

  def build(self, input_shape):
      """构建模型各层"""
      self.fc1.build(input_shape)
      sp1 = self.fc1.compute_output_shape(input_shape)

      self.fc2.build(sp1)
      sp2 = self.fc2.compute_output_shape(sp1)
      self.fc3.build(sp1)
      sp3 = self.fc3.compute_output_shape(sp1)

      self.fc4.build([None, z_dim])
      sp4 = self.fc4.compute_output_shape([None, z_dim])
      self.fc5.build(sp4)
      sp1 = self.fc5.compute_output_shape(sp4)
      self.built = True
      super(VAE, self).build(input_shape)

model = VAE()
model.build(input_shape=(None, 784))
model.summary()


opt = optimizers.Adam(learning_rate=1e-3)

for epoch in range(100):

  for step, x in enumerate(train_db):

    x = tf.reshape(x, [-1, 784])

    with tf.GradientTape() as tape:
      x_rec_logits , mu, log_var = model(x)

      rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits='True')
      rec_loss = tf.reduce_mean(rec_loss)

      # compute kl divergence (mu, var) - N(0, 1)
      # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
      kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
      kl_div = tf.reduce_mean(kl_div) / x.shape[0]

      loss = rec_loss + 1. * kl_div

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    if step % 100 == 0:
      print(epoch, step, 'kl div:',float(kl_div), 'rec_loss', float(rec_loss))


    # evaluation
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_image(x_hat, '11-AutoEncoder/vae_imgs/sampled_epoch_%d.png'%epoch)

    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _ , _ = model(x)
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_image(x_hat, '11-AutoEncoder/vae_imgs/rec_epoch_%d.png'%epoch)


