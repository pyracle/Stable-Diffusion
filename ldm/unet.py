import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from train_utils.noise_scheduler import Scheduler
from variational_autoencoder import VariationalAutoEncoder
from train_utils.download_flickr_pipeline import DataLoader
from base import AbstractConfig, ResNetBlock, TextEncoder, LearningRateSchedule


class Attention(AbstractConfig):
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, query, context, **kwargs):
        attn_output, attn_scores = self.mha(
            query=query,
            key=context,
            value=context,
            return_attention_scores=True,
            training=kwargs.get('training', False),
            use_causal_mask=kwargs.get('use_causal_mask', False)
        )
        self.last_attn_scores = attn_scores
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
        ) for _ in range(8)]
        self.resnet_blocks = [ResNetBlock(64 * 2 ** i, dropout_rate) for i in range(3)]
        self.max_pool = [keras.layers.MaxPooling2D() for _ in range(3)]
        self.layer_norm = [keras.layers.LayerNormalization() for _ in range(3)]
        self.up_sampling = [keras.layers.UpSampling2D() for _ in range(3)]
        self.conv_output = keras.layers.Conv2D(4, 3, padding='same')
        self.repetitions = repetitions

    def call(self, inputs, training=False):
        img, text, time_step = inputs
        time_step = self.dense_time(time_step, training=training)
        time_step = self.layer_norm[0](time_step, training=training)

        x_64 = self.unet_blocks[0](img, time_step, training=training)
        x_64 = self.attention_layers[0](x_64, text, training=training)
        x = self.max_pool[0](x_64, training=training)
        x = self.dropout_layers[0](x, training=training)
        x_32 = self.unet_blocks[1](x, time_step, training=training)
        x_32 = self.attention_layers[1](x_32, text, training=training)
        x = self.max_pool[1](x_32, training=training)
        x = self.dropout_layers[1](x, training=training)
        x_16 = self.unet_blocks[2](x, time_step, training=training)
        x_16 = self.attention_layers[2](x_16, text, training=training)
        x = self.max_pool[2](x_16, training=training)
        x = self.dropout_layers[2](x, training=training)
        x_8 = self.unet_blocks[3](x, time_step, training=training)
        x_8 = self.attention_layers[3](x_8, text, training=training)
        x_8 = self.dropout_layers[3](x_8, training=training)

        for block in self.resnet_blocks:
            x = block(x, training=training)

        x = tf.concat([x, x_8], -1)
        x = self.unet_blocks[4](x, time_step, training=training)
        x = self.attention_layers[4](x, text, training=training)
        x = self.up_sampling[0](x)
        x = self.dropout_layers[4](x, training=training)
        x = tf.concat([x, x_16], -1)
        x = self.unet_blocks[5](x, time_step, training=training)
        x = self.attention_layers[5](x, text, training=training)
        x = self.up_sampling[1](x)
        x = self.dropout_layers[5](x, training=training)
        x = tf.concat([x, x_32], -1)
        x = self.unet_blocks[6](x, time_step, training=training)
        x = self.attention_layers[6](x, text, training=training)
        x = self.up_sampling[2](x)
        x = self.dropout_layers[6](x, training=training)
        x = tf.concat([x, x_64], -1)
        x = self.unet_blocks[7](x, time_step, training=training)
        x = self.attention_layers[7](x, text, training=training)
        x = self.dropout_layers[7](x, training=training)
        return self.conv_output(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'repetitions': self.repetitions
        })
        return config


class TrainUNet(keras.Model):
    def __init__(self,
                 text_encoder: TextEncoder,
                 vae: VariationalAutoEncoder,
                 unet: UNet,
                 scheduler: Scheduler,
                 batch_size: int = 32,
                 **kwargs):
        super(TrainUNet, self).__init__(**kwargs)
        self.text_encoder = text_encoder
        self.text_encoder.trainable = False
        self.vae = vae
        self.vae.trainable = False
        self.unet = unet
        self.scheduler = scheduler
        self.batch_size = batch_size

    @tf.function
    def train_step(self, data):
        image_batch, text_batch = data
        with tf.GradientTape() as tape:
            loss = self.compute_loss(image_batch, text_batch, training=True)

        grads = tape.gradient(loss['loss'], self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.unet.trainable_variables))
        return loss

    @tf.function
    def test_step(self, data):
        return self.compute_loss(data[0], data[1])

    @tf.function
    def compute_loss(self, image_batch, text_batch, training=False):
        image_batch = self.vae.encoder(image_batch)[0]
        batch_size = text_batch.shape[0]
        text_batch = self.text_encoder(text_batch, training=training)
        time_step = self.generate_time_step(batch_size)
        label_img_batch, feature_img_batch = self.scheduler(
            image_batch,
            time_step
        )
        logits = self.unet(
            (feature_img_batch, text_batch, time_step),
            training=training
        )
        loss = tf.reduce_mean(
            tf.square(
                label_img_batch - logits
            )
        )
        return {
            'loss': loss
        }

    @tf.function
    def generate_time_step(self, batch_size):
        return tf.random.uniform((batch_size, 1), 0, self.unet.repetitions - 1, tf.int32)


class TrainUnetCallback(keras.callbacks.Callback):
    def __init__(self,
                 val_data: tf.data.Dataset,
                 unet_checkpoint_path: str,
                 epochs: int = 10):
        super().__init__()
        epoch_addition = '/saved-model-{epoch:'f'{len(str(epochs))}''.0f}'
        self.unet_checkpoint_path = unet_checkpoint_path + epoch_addition

        for images, descriptions in val_data:
            self.images = tf.slice(images, [4, 0, 0, 0], [2, -1, -1, -1])
            self.descriptions = tf.slice(descriptions, [4], [2])

    def on_train_begin(self, logs=None):
        vae_predictions = self.model.vae(self.images)
        return self.plot_images(vae_predictions)

    def on_epoch_end(self, epoch, logs=None):
        text = self.model.text_encoder(self.descriptions)
        image = tf.random.normal((2, 64, 64, 4))
        for time_step in range(self.model.unet.repetitions):
            time_step = tf.constant(2 * [time_step])
            time_step = tf.expand_dims(time_step, -1)
            image -= self.model.unet((image, text, time_step))
        predictions = self.model.vae.decoder(image)
        self.plot_images(predictions)
        self.model.unet.save_weights(
            self.unet_checkpoint_path.format(epoch=epoch + 1).replace(' ', '0')
        )

    @staticmethod
    def plot_images(images):
        plt.figure(figsize=(4, 2), dpi=200)
        for idx, image in enumerate(images):
            plt.subplot(1, 2, idx + 1)
            image = image / 2 + .5
            plt.imshow(image)
            plt.axis('off')
        plt.show()


def print_model_summary():
    batch_size = 1
    img = tf.random.normal((batch_size, 64, 64, 4))
    text = tf.random.normal((batch_size, 64, 4, 384))
    time_step = tf.random.uniform((batch_size, 1), 0, 49, tf.int32)
    unet = UNet(units=256,
                num_attention_heads=8)
    unet((img, text, time_step))
    print(unet.summary())


def train():
    batch_size = 32
    epochs = 100
    units = 256
    train_ds, val_ds = DataLoader(
        batch_size=batch_size,
        data_dir='../train_utils/data/flickr8k'
    )(mode='both')
    unet = UNet(units=units,
                num_attention_heads=8)
    text_encoder = TextEncoder()
    vae = tf.saved_model.load('checkpoints/vae')
    unet_train = TrainUNet(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=Scheduler(50),
        batch_size=batch_size
    )
    unet_train_callback = TrainUnetCallback(
        unet_checkpoint_path='checkpoints/unet',
        val_data=val_ds.take(1),
        epochs=epochs
    )
    lr_schedule = LearningRateSchedule(units)
    unet_train.compile(optimizer=keras.optimizers.Adam(lr_schedule))
    unet_train.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[unet_train_callback]
    )


if __name__ == '__main__':
    print_model_summary()
    train()
