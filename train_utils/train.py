import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from ldm.base import UNet, TextEncoder, VariationalAutoEncoder
from noise_scheduler import Scheduler
from download_flickr_pipeline import DataLoader


class LearningRateSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_units, warmup_steps=5000):
        super().__init__()
        self.units = tf.cast(max_units, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.units) * tf.math.minimum(arg1, arg2)


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


def train_vae(vae: VariationalAutoEncoder):
    epochs = 100
    batch_size = 32
    lr_schedule = LearningRateSchedule(512)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/vae/saved-model-{epoch:03d}'
    )
    train_images, val_images = DataLoader(
        batch_size=batch_size,
        data_dir='../train_utils/data/flickr8k'
    )(mode='image')
    vae.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=keras.losses.MeanSquaredError()
    )
    vae.fit(
        train_images,
        epochs=epochs,
        validation_data=val_images,
        callbacks=[model_checkpoint, PlotImages(val_images.take(1))]
    )


def train_unet(unet: UNet):
    batch_size = 32
    epochs = 100
    units = 256
    train_ds, val_ds = DataLoader(
        batch_size=batch_size,
        data_dir='../train_utils/data/flickr8k'
    )(mode='both')
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
