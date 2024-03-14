import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from typing import Callable
from base_config import AbstractConfig
from train_utils.download_flickr_pipeline import DataLoader


class ResNetBlock(AbstractConfig):
    def __init__(self,
                 max_filters: int,
                 dropout_rate: float,
                 relu: Callable):
        super(ResNetBlock, self).__init__()
        
        self.sequential = keras.Sequential([
            keras.layers.Conv2D(round(max_filters / 4), (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            relu(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Conv2D(round(max_filters / 2), (4, 4), padding='same'),
            keras.layers.BatchNormalization(),
            relu(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Conv2D(max_filters, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            relu(),
        ])
        self.skip = keras.Sequential([
            keras.layers.Conv2D(max_filters, (1, 1), padding='same'),
            keras.layers.BatchNormalization(),
            relu()
        ])

    def call(self, inputs, training=False):
        x = self.sequential(inputs, training=training)
        x_skip = self.skip(inputs, training=training)
        return x + x_skip


class AbstractSequential(AbstractConfig):
    def __init__(self,
                 convolutional_layer: Callable,
                 filters: int,
                 activation_func=None,
                 dropout_rate: float = .1,
                 **kwargs):
        super(AbstractSequential, self).__init__(**kwargs)
        
        if convolutional_layer == keras.layers.Conv2D:
            relu = keras.layers.ReLU
            pooling_or_up_sampling = keras.layers.AveragePooling2D
        else:
            relu = keras.layers.LeakyReLU
            pooling_or_up_sampling = keras.layers.UpSampling2D
            
        self.sequential = keras.Sequential([
            ResNetBlock(256, dropout_rate, relu),
            keras.layers.Dropout(dropout_rate),
            ResNetBlock(512, dropout_rate, relu),
            pooling_or_up_sampling(),  # (b, 128, 128, 512)
            keras.layers.Dropout(dropout_rate),
            ResNetBlock(256, dropout_rate, relu),
            keras.layers.Dropout(dropout_rate),
            ResNetBlock(512, dropout_rate, relu),
            pooling_or_up_sampling(),  # (b, 64, 64, 512)
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(filters, activation=activation_func)
        ])


class Encoder(AbstractConfig):
    def __init__(self,
                 latent_dim: int,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.sequential = AbstractSequential(
            convolutional_layer=keras.layers.Conv2D,
            filters=latent_dim
        ).sequential
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_log_var = keras.layers.Dense(latent_dim)

    def call(self, inputs, training=False):
        x = self.sequential(inputs, training=training)
        z_mean = self.dense_mean(x, training=training)
        z_log_var = self.dense_log_var(x, training=training)
        z = self.sampling(z_mean, z_log_var)

        assert (z.get_shape() == z_mean.get_shape() == z_log_var.get_shape()
                == (inputs.get_shape()[0], 64, 64, self.latent_dim))
        return z, z_mean, z_log_var

    @tf.function
    def sampling(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        epsilon = keras.backend.random_normal(shape=(batch_size, 64, 64, self.latent_dim))
        z = tf.exp(.5 * z_log_var) * epsilon + z_mean
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim
        })
        return config


class Decoder(AbstractConfig):
    def __init__(self,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense = keras.layers.Dense(16)
        self.reshape = keras.layers.Reshape((64, 64, 16))
        self.sequential = AbstractSequential(
            convolutional_layer=keras.layers.Conv2DTranspose,
            filters=3,
            activation_func='tanh'
        ).sequential

    def call(self, inputs, training=False):
        x = self.dense(inputs, training=training)
        x = self.reshape(x)
        x = self.sequential(x, training=training)

        assert x.get_shape() == (inputs.get_shape()[0], 256, 256, 3)
        return x


class VariationalAutoEncoder(keras.Model,
                             AbstractConfig):
    def __init__(self,
                 latent_dim: int,
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    def call(self, inputs, training=False):
        z = self.encoder(inputs, training=training)[0]
        x = self.decoder(z, training=training)
        return x

    @tf.function
    def train_step(self, data) -> dict:
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data, training=True)

        grads = tape.gradient(loss['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    @tf.function
    def test_step(self, data) -> dict:
        return self.compute_loss(data)

    @tf.function
    def compute_loss(self, data, training=False):
        z, z_mean, z_log_var = self.encoder(data, training=training)
        reconstructed = self.decoder(z, training=training)
        mse = tf.reduce_mean(tf.square(
            data - reconstructed
        ))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(kl_loss, axis=-1)
        )
        total_loss = mse + kl_loss
        return {
            'loss': total_loss,
            'mse': mse,
            'kl_loss': kl_loss
        }

    def get_config(self):
        config = super(VariationalAutoEncoder, self).get_config()
        config.update(
            {'latent_dim': self.encoder.latent_dim}
        )
        return config


class PlotImages(keras.callbacks.Callback):
    def __init__(self,
                 validation_data):
        super().__init__()
        data = validation_data.take(1)
        for images in data:
            self.validation_data = tf.slice(images, [0, 0, 0, 0], [4, -1, -1, -1])

    def on_train_begin(self, logs=None):
        self.plot_images(self.validation_data)

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.validation_data)
        self.plot_images(predictions)

    @staticmethod
    def plot_images(images):
        plt.figure(figsize=(4, 4), dpi=200)
        for idx, image in enumerate(images):
            plt.subplot(2, 2, idx + 1)
            image = image / 2 + .5
            plt.axis('off')
            plt.imshow(image)
        plt.show()


def print_model_summary():
    latent_dim = 4
    vae = VariationalAutoEncoder(latent_dim)
    sample = tf.random.uniform(shape=(1, 256, 256, 3),
                               maxval=1.0,
                               dtype=tf.float32)
    vae(sample)
    print(vae.summary())
    return vae


def train(vae: VariationalAutoEncoder):
    epochs = 100
    batch_size = 32
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/vae/saved-model-{epoch:03d}'
    )
    train_images, val_images = DataLoader(
        batch_size=batch_size,
        data_dir='train_utils/data/flickr8k'
    )(mode='image')
    vae.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True)
    )
    vae.fit(
        train_images,
        epochs=epochs,
        validation_data=val_images,
        callbacks=[model_checkpoint, PlotImages(val_images.take(1))]
    )


if __name__ == '__main__':
    vae = print_model_summary()
    train(vae)
