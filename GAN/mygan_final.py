import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

ds = tfds.load('fashion_mnist', split='train')
ds.as_numpy_iterator().next().keys()

iterator = ds.as_numpy_iterator()


fig, ax = plt.subplots(1, 5, figsize=(20, 20))
for i in range(5):
  ax[i].imshow(iterator.next()['image'])


#scale krne ke liye
def scale(data):
    image = data['image']
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image

#ds preprocess
ds = ds.map(scale)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(256)
ds = ds.prefetch(tf.data.AUTOTUNE)

#building generator
def generator():
  model = Sequential()

  model.add(Dense (7*7*128, input_dim= 128))
  model.add(LeakyReLU(0.2))
  model.add(Reshape((7,7,128)))

  model.add(UpSampling2D()) #14*14*128
  model.add(Conv2D(128, kernel_size=5, padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(UpSampling2D()) #28*28*128
  model.add(Conv2D(128, kernel_size=5, padding='same'))
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(128, kernel_size=4, padding='same')) #learning
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(128, kernel_size=4, padding='same')) #learning
  model.add(LeakyReLU(0.2))

  model.add(Conv2D(1, kernel_size=4, padding='same', activation='sigmoid')) #28*28*1

  return model

img = generator().predict(np.random.randn(4,128,1))
img.shape

fig, ax = plt.subplots(1, 4, figsize=(20, 20))
for i, display_img in enumerate(img):  # Change variable name here
    ax[i].imshow(np.squeeze(display_img))

#building discriminator
def discriminator():
  model = Sequential()

  model.add(Conv2D(32, kernel_size=5, input_shape=(28,28,1)))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(64, kernel_size=5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(128, kernel_size=5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(256, kernel_size=5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4))

  model.add(Flatten())
  model.add(Dropout(0.4))
  model.add(Dense(1, activation='sigmoid'))

  return model

discriminator().predict(img)

#for single picture
img = img[0]
img = np.expand_dims(img, axis=0)
discriminator().predict(img)

class myGan(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = generator()
        self.d = discriminator()

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        real_images = batch
        batch_size = tf.shape(batch)[0]
        noise = tf.random.normal((batch_size, 128))
        fake_images = self.g(noise, training=False)

        #discriminator training
        with tf.GradientTape() as d_tape:
            yhat_real = self.d(real_images, training=True)
            yhat_fake = self.d(fake_images, training=True)

            
            real_labels = tf.ones_like(yhat_real) * 0.9
            fake_labels = tf.zeros_like(yhat_fake)
            total_d_loss = self.d_loss(real_labels, yhat_real) + self.d_loss(fake_labels, yhat_fake)

        d_grad = d_tape.gradient(total_d_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.d.trainable_variables))

        #generator training
        with tf.GradientTape() as g_tape:
            noise = tf.random.normal((batch_size, 128))
            gen_images = self.g(noise, training=True)
            predicted_labels = self.d(gen_images, training=False)
            total_g_loss = self.g_loss(tf.ones_like(predicted_labels), predicted_labels)

        g_grad = g_tape.gradient(total_g_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.g.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

g_opt = Adam(learning_rate=0.0001, beta_1=0.5)
d_opt = Adam(learning_rate=0.00005, beta_1=0.5)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

mygan = myGan(generator, discriminator)
mygan.compile(g_opt, d_opt, g_loss, d_loss)

class ModelMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=5, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        os.makedirs("images", exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):

        latent_vec = tf.random.normal((self.num_img, self.latent_dim))
        gen_img = self.model.g(latent_vec)
        gen_img = (gen_img * 255).numpy()


        fig, ax = plt.subplots(1, self.num_img, figsize=(20, 20))
        for i in range(self.num_img):
            ax[i].imshow(gen_img[i].squeeze(), cmap='gray')
            ax[i].axis('off')
        plt.show()


        self.model.g.save(f"models/generator_epoch_{epoch + 1}.h5")
        self.model.d.save(f"models/discriminator_epoch_{epoch + 1}.h5")


monitor = ModelMonitor(num_img=5) 
history = mygan.fit(ds, epochs=400, callbacks=[monitor])

from tensorflow.keras.models import load_model

generator = load_model('/kaggle/input/testgan/keras/default/1/generator_epoch_204.h5')

num_img = 20 
random_latent_vectors = tf.random.normal((num_img, 128))
imgs = generator(random_latent_vectors)

fig, ax = plt.subplots(4,4, figsize=(10,10))
for r in range(4): 
    for c in range(4): 
        ax[r][c].imshow(imgs[(r+1)*(c+1)-1], cmap='gray')
        ax[r][c].axis('off')
plt.show()
