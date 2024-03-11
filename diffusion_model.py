import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
from base_config import AbstractConfig
from variational_autoencoder import VariationalAutoEncoder
from train_utils.noise_scheduler import Scheduler
from train_utils.download_flickr_pipeline import DataLoader


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


class UNetBlock(AbstractConfig):
    def __init__(self,
                 **kwargs):
        super(UNetBlock, self).__init__(**kwargs)
        self.conv = keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.dense_time = keras.layers.Dense(256)
        self.reshape_time = keras.layers.Reshape((1, 1, 256))
        self.conv_output = keras.layers.Conv2D(256, 4, padding='same')
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, img, time_step, training=False):
        img = self.conv(img, training=training)
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
                 dropout_rate: float,
                 repetitions: int = 50,
                 **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.dense_time = keras.layers.Dense(256)
        self.unet_blocks = [UNetBlock() for _ in range(8)]
        self.dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in range(8)]
        self.max_pool = [keras.layers.MaxPooling2D() for _ in range(3)]
        self.flatten = keras.layers.Flatten()
        self.dense_mlp_a = keras.layers.Dense(256)
        self.dense_mlp_b = keras.layers.Dense(8 * 8 * 16)
        self.reshape = keras.layers.Reshape((8, 8, 16))
        self.layer_norm = [keras.layers.LayerNormalization() for _ in range(3)]
        self.up_sampling = [keras.layers.UpSampling2D() for _ in range(3)]
        self.conv_output = keras.layers.Conv2D(4, 3, padding='same')
        self.repetitions = repetitions

    def call(self, inputs, training=False):
        img, text, time_step = inputs
        img = tf.matmul(img, text)
        time_step = self.dense_time(time_step, training=training)
        time_step = self.layer_norm[0](time_step, training=training)

        x_64 = self.unet_blocks[0](img, time_step, training=training)
        x = self.max_pool[0](x_64, training=training)
        x = self.dropout_layers[0](x, training=training)

        x_32 = self.unet_blocks[1](x, time_step, training=training)
        x = self.max_pool[1](x_32, training=training)
        x = self.dropout_layers[1](x, training=training)
        x_16 = self.unet_blocks[2](x, time_step, training=training)
        x = self.max_pool[2](x_16, training=training)
        x = self.dropout_layers[2](x, training=training)
        x_8 = self.unet_blocks[3](x, time_step, training=training)
        x_8 = self.dropout_layers[3](x_8, training=training)

        assert x_8.get_shape()[1:] == (8, 8, 256)

        x = self.flatten(x_8)
        x = tf.concat([x, time_step], axis=-1)

        x = self.dense_mlp_a(x, training=training)
        x = self.layer_norm[1](x, training=training)
        x = tf.nn.relu(x)
        x = self.dense_mlp_b(x, training=training)
        x = self.layer_norm[2](x, training=training)
        x = tf.nn.relu(x)
        x = self.reshape(x)

        x = tf.concat([x, x_8], -1)
        x = self.unet_blocks[4](x, time_step, training=training)
        x = self.up_sampling[0](x)
        x = self.dropout_layers[4](x, training=training)
        x = tf.concat([x, x_16], -1)
        x = self.unet_blocks[5](x, time_step, training=training)
        x = self.up_sampling[1](x)
        x = self.dropout_layers[5](x, training=training)
        x = tf.concat([x, x_32], -1)
        x = self.unet_blocks[6](x, time_step, training=training)
        x = self.up_sampling[2](x)
        x = self.dropout_layers[6](x, training=training)
        x = tf.concat([x, x_64], -1)
        x = self.unet_blocks[7](x, time_step, training=training)
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
        self.vae = vae
        self.vae.trainable = False
        self.unet = unet
        self.scheduler = scheduler
        self.batch_size = batch_size

    @tf.function
    def train_step(self, data):
        image_batch, text_batch = data
        with tf.GradientTape() as unet_tape, tf.GradientTape() as text_encoder_tape:
            loss = self.compute_loss(image_batch, text_batch, training=True)

        grads = unet_tape.gradient(loss['loss'], self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_variables))
        grads = text_encoder_tape.gradient(loss['loss'], self.text_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.text_encoder.trainable_variables))
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
            (text_batch, feature_img_batch, time_step),
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
                 text_encoder_checkpoint_path: str,
                 epochs: int = 10):
        super().__init__()
        epoch_addition =  '/saved-model-{epoch:'f'{len(str(epochs))}''.0f}'
        self.unet_checkpoint_path = unet_checkpoint_path + epoch_addition
        self.text_encoder_checkpoint_path = text_encoder_checkpoint_path + epoch_addition

        for images, descriptions in val_data:
            self.images = tf.slice(images, [0, 0, 0, 0], [4, -1, -1, -1])
            self.descriptions = tf.slice(descriptions, [0], [4])

    def on_train_begin(self, logs=None):
        vae_predictions = self.model.vae(self.images)
        return self.plot_images(vae_predictions)

    def on_epoch_end(self, epoch, logs=None):
        text = self.model.text_encoder(self.descriptions)
        image = tf.random.normal((4, 64, 64, 4))
        for time_step in range(self.model.unet.repetitions):
            time_step = tf.constant(4 * [time_step])
            time_step = tf.expand_dims(time_step, -1)
            image = self.model.unet((image, text, time_step))
        predictions = self.model.vae.decoder(image)
        self.plot_images(predictions)

        epoch += 1
        self.model.unet.save_weights(self.unet_checkpoint_path.format(epoch=epoch).replace(' ', '0'))
        self.model.text_encoder.save(self.text_encoder_checkpoint_path.format(epoch=epoch).replace(' ', '0'))

    @staticmethod
    def plot_images(images):
        plt.figure(figsize=(4, 4), dpi=200)
        for idx, image in enumerate(images):
            plt.subplot(2, 2, idx + 1)
            image = image / 2 + .5
            plt.imshow(image)
            plt.axis('off')
        plt.show()


def print_model_summary():
    batch_size = 32
    img = tf.random.normal((batch_size, 64, 64, 4))
    text = tf.random.normal((batch_size, 64, 4, 384))
    time_step = tf.random.uniform((batch_size, 1), 0, 49, tf.int32)
    dropout_rate = 0.1
    unet = UNet(dropout_rate)
    unet((img, text, time_step))
    print(unet.summary())


def train():
    batch_size = 32
    unet_repetitions = 50
    epochs = 100
    train_ds, val_ds = DataLoader(
        batch_size=batch_size,
        data_dir='train_utils/data/flickr8k'
    )(mode='both')
    unet = UNet(dropout_rate=.1,
                repetitions=unet_repetitions)
    text_encoder = TextEncoder()
    vae = tf.saved_model.load('checkpoints/vae/saved-model-080')
    unet_train = TrainUNet(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=Scheduler(50),
        batch_size=batch_size
    )
    unet_train_callback = TrainUnetCallback(
        unet_checkpoint_path='checkpoints/unet',
        text_encoder_checkpoint_path='checkpoints/text_encoder',
        val_data=val_ds.take(1),
        epochs=epochs
    )
    unet_train.compile(optimizer=keras.optimizers.Adam())
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
