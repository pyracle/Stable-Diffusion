import tensorflow as tf
from tensorflow import keras
from diffusion_model import UNet
from base_config import AbstractConfig
from variational_autoencoder import Decoder
from train_utils.noise_scheduler import Scheduler


class StableDiffusion(keras.Model,
                      AbstractConfig):
    def __init__(self,
                 unet_repetitions: int,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super(StableDiffusion, self).__init__(**kwargs)
        self.unet_repetitions = unet_repetitions
        self.noise_scheduler = Scheduler(
            unet_repetitions
        )
        self.diffusion_model = UNet(
            dropout_rate
        )
        self.text_encoder = self.diffusion_model.text_encoder
        self.text_encoder.trainable = False
        self.image_decoder = Decoder(
            dropout_rate=dropout_rate
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        text = self.text_encoder(inputs, training=training)

        x = self.noise_scheduler(batch_size)
        for time_step in range(self.unet_repetitions):
            time_step = int(batch_size) * [time_step]
            time_step = tf.constant(time_step)[:, tf.newaxis]
            x = self.diffusion_model((x, text, time_step), training=training)

        logits = self.image_decoder(x, training=training)
        return logits


def print_model_summary():
    stable_diffusion_model = StableDiffusion(
        unet_repetitions=50,
        dropout_rate=.1
    )
    batch_size = 32
    test_sample = tf.random.uniform((batch_size, 128), 0, 5000, tf.int32)
    stable_diffusion_model(test_sample)
    print(stable_diffusion_model.summary())


if __name__ == '__main__':
    print_model_summary()
