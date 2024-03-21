import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
from typing import Callable


class AbstractConfig(keras.layers.Layer):
    def get_config(self):
        return super().get_config()


class ResNetBlock(AbstractConfig):
    def __init__(self,
                 max_filters: int,
                 dropout_rate: float,
                 relu=keras.layers.ReLU):
        super(ResNetBlock, self).__init__()
        self.sequential = keras.Sequential([
            keras.layers.Conv2D(round(max_filters / 4), (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            relu(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Conv2D(round(max_filters / 2), (5, 5), padding='same'),
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


class TextEncoder(keras.Model,
                  AbstractConfig):
    def __init__(self,
                 **kwargs):
        super(TextEncoder, self).__init__(**kwargs)

        self.preprocessor = hub.KerasLayer(
            'https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/'
            'variations/en-uncased-preprocess/versions/3'
        )
        self.text_encoder = hub.KerasLayer(
            'https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/'
            'variations/bert-en-uncased-l-2-h-768-a-12/versions/2',
            trainable=True
        )
        self.text_encoder.trainable = False
        self.reshape = keras.layers.Reshape((64, 4, 384))

    def call(self, inputs, training=False):
        x = self.preprocessor(inputs)
        x = self.text_encoder(x)['sequence_output']
        return self.reshape(x)


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
        return self.sequential(x, training=training)


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
        return self.decoder(z, training=training)

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
        reconstruction_loss = self.loss(data, reconstructed)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(kl_loss, axis=-1)
        )
        total_loss = reconstruction_loss + kl_loss
        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss
        }

    def get_config(self):
        config = super(VariationalAutoEncoder, self).get_config()
        config.update(
            {'latent_dim': self.encoder.latent_dim}
        )
        return config


class Attention(AbstractConfig):
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, query, context, **kwargs):
        attn_output, self.last_attn_scores = self.mha(
            query=query,
            key=context,
            value=context,
            return_attention_scores=True,
            training=kwargs.get('training', False),
            use_causal_mask=kwargs.get('use_causal_mask', False)
        )
        x = attn_output + query
        return self.layer_norm(x)


class UNetBlock(AbstractConfig):
    def __init__(self,
                 units: int,
                 dropout_rate: float,
                 **kwargs):
        super(UNetBlock, self).__init__(**kwargs)
        self.res_block = ResNetBlock(units, dropout_rate)
        self.dense_time = keras.layers.Dense(units)
        self.reshape_time = keras.layers.Reshape((1, 1, units))
        self.conv_output = keras.layers.Conv2D(256, 5, padding='same')
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, img, time_step, training=False):
        img = self.res_block(img, training=training)
        time_step = self.dense_time(time_step, training=training)
        time_step = self.reshape_time(time_step)
        img_time = img + time_step
        output = self.conv_output(img, training=training)
        output += img_time
        output = self.layer_norm(output, training=training)
        return tf.nn.relu(output)


class UNet(keras.Model,
           AbstractConfig):
    def __init__(self,
                 units: int,
                 num_attention_heads: int,
                 dropout_rate: float = .1,
                 repetitions: int = 50,
                 **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.dense_time = keras.layers.Dense(units)
        self.unet_blocks = [UNetBlock(units, dropout_rate) for _ in range(8)]
        self.dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in range(8)]
        self.attention_layers = [Attention(
            num_heads=num_attention_heads,
            key_dim=units
        ) for _ in range(11)]
        self.resnet_blocks = [ResNetBlock(256, dropout_rate) for _ in range(3)]
        self.max_pool = [keras.layers.MaxPooling2D() for _ in range(3)]
        self.layer_norm = [keras.layers.LayerNormalization() for _ in range(3)]
        self.up_sampling = [keras.layers.UpSampling2D() for _ in range(3)]
        self.concat_layers = [keras.layers.Concatenate(axis=-1) for _ in range(4)]
        self.conv_output = keras.layers.Conv2D(4, 3, padding='same')
        self.repetitions = repetitions

    def call(self, inputs, training=False):
        img, text, time_step = inputs
        time_step = self.dense_time(time_step, training=training)
        time_step = self.layer_norm[0](time_step, training=training)

        x_64 = self.unet_blocks[0](img, time_step, training=training)  # (64, 64, units)
        x_64 = self.attention_layers[0](x_64, text, training=training)
        x = self.max_pool[0](x_64, training=training)
        x = self.dropout_layers[0](x, training=training)
        x_32 = self.unet_blocks[1](x, time_step, training=training)  # (32, 32, units)
        x_32 = self.attention_layers[1](x_32, text, training=training)
        x = self.max_pool[1](x_32, training=training)
        x = self.dropout_layers[1](x, training=training)
        x_16 = self.unet_blocks[2](x, time_step, training=training)  # (16, 16, units)
        x_16 = self.attention_layers[2](x_16, text, training=training)
        x = self.max_pool[2](x_16, training=training)
        x = self.dropout_layers[2](x, training=training)
        x_8 = self.unet_blocks[3](x, time_step, training=training)  # (8, 8, units)
        x_8 = self.attention_layers[3](x_8, text, training=training)
        x_8 = self.dropout_layers[3](x_8, training=training)

        for block_idx in range(len(self.resnet_blocks)):
            x = self.resnet_blocks[block_idx](x, training=training)  # (8, 8, units)
            x = self.attention_layers[4 + block_idx](x, x, training=training)

        x = self.concat_layers[0]([x, x_8])
        x = self.unet_blocks[4](x, time_step, training=training)  # (8, 8, units)
        x = self.attention_layers[7](x, text, training=training)
        x = self.up_sampling[0](x)
        x = self.dropout_layers[4](x, training=training)
        x = self.concat_layers[1]([x, x_16])
        x = self.unet_blocks[5](x, time_step, training=training)  # (16, 16, units)
        x = self.attention_layers[8](x, text, training=training)
        x = self.up_sampling[1](x)
        x = self.dropout_layers[5](x, training=training)
        x = self.concat_layers[2]([x, x_32])
        x = self.unet_blocks[6](x, time_step, training=training)  # (32, 32, units)
        x = self.attention_layers[9](x, text, training=training)
        x = self.up_sampling[2](x)
        x = self.dropout_layers[6](x, training=training)
        x = self.concat_layers[3]([x, x_64])
        x = self.unet_blocks[7](x, time_step, training=training)  # (64, 64, units)
        x = self.attention_layers[10](x, text, training=training)
        x = self.dropout_layers[7](x, training=training)
        return self.conv_output(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'repetitions': self.repetitions
        })
        return config


def print_vae_summary():
    latent_dim = 4
    vae = VariationalAutoEncoder(latent_dim)
    sample = tf.random.uniform(shape=(1, 256, 256, 3),
                               maxval=1.0,
                               dtype=tf.float32)
    vae(sample)
    print(vae.summary())


def print_unet_summary():
    batch_size = 1
    img_batch = tf.random.normal((batch_size, 64, 64, 4))
    text_batch = tf.random.normal((batch_size, 64, 4, 384))
    time_step_batch = tf.random.uniform((batch_size, 1), 0, 49, tf.int32)
    unet = UNet(units=256,
                num_attention_heads=8)
    unet((img_batch, text_batch, time_step_batch))
    print(unet.summary())


if __name__ == '__main__':
    print_vae_summary()
    print_unet_summary()
