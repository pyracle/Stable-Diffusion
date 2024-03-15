"""
Complete Latent Diffusion model
"""

import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from tensorflow import keras
from base_config import AbstractConfig
from variational_autoencoder import Decoder
from diffusion_model import UNet, TextEncoder
from train_utils.download_flickr_pipeline import DataLoader


class StableDiffusion(keras.Model,
                      AbstractConfig):
    def __init__(self,
                 unet: UNet,
                 image_decoder: Decoder,
                 dropout_rate: float = 0.1,
                 text_encoder=TextEncoder(),
                 **kwargs):
        super(StableDiffusion, self).__init__(**kwargs)
        self.unet = unet
        self.text_encoder = text_encoder
        self.text_encoder.trainable = False
        self.image_decoder = image_decoder

    def call(self, inputs, training=False):
        batch_size = inputs.get_shape()[0]
        text = self.text_encoder(inputs, training=False)
        image = tf.random.normal((batch_size, 64, 64, 4))
        for time_step in range(self.unet.repetitions):
            time_step = tf.constant(batch_size * [time_step])
            time_step = tf.expand_dims(time_step, -1)
            image -= self.unet((image, text, time_step), training=training)
        return self.image_decoder(image, training=False)


def print_model_summary():
    unet = UNet(dff=64,
                d_model=256,
                num_attention_heads=8)
    vae = tf.saved_model.load('checkpoints/vae')
    stable_diffusion_model = StableDiffusion(
        unet=unet,
        image_decoder=vae.decoder
    )
    batch_size = 8
    test_sample = DataLoader(
        batch_size=batch_size,
        data_dir='train_utils/data/flickr8k'
    )(mode='text')[0].take(1)
    for test_sample in test_sample:
        break
    stable_diffusion_model(test_sample)
    print(stable_diffusion_model.summary())


if __name__ == '__main__':
    print_model_summary()
