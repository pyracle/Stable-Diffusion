import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


class Scheduler(keras.layers.Layer):
    def __init__(self,
                 repetitions: int,
                 **kwargs):
        super(Scheduler, self).__init__(**kwargs)
        self.repetitions = repetitions + 1

    def call(self, inputs, time_step):
        image_shape = (inputs.get_shape()[0], 64, 64, 4)
        time_step = tf.slice(time_step, [0, 0], [1, 1])

        label_img_fraction = tf.cast((time_step + 1) / self.repetitions, tf.float32)
        feature_img_fraction = tf.cast(time_step / self.repetitions, tf.float32)
        label_img = self.add_noise_to_images(
            inputs,
            image_shape,
            label_img_fraction
        )
        feature_img = self.add_noise_to_images(
            inputs,
            image_shape,
            feature_img_fraction
        )

        assert label_img.get_shape() == image_shape
        assert feature_img.get_shape() == image_shape

        return label_img, feature_img
    
    @staticmethod
    def add_noise_to_images(inputs, img_shape, img_fraction):
        noise_fraction = 1 - img_fraction
        noise_fraction *= inputs
        img_fraction *= tf.random.normal(img_shape)
        img_batch = img_fraction + noise_fraction
        return tf.cast(img_batch, tf.float32)


def test_scheduler():
    batch_size = 8
    image_batch = tf.random.uniform(
        shape=(batch_size, 64, 64, 4),
        minval=-1,
        maxval=1
    )
    scheduler = Scheduler(50)
    time_step = tf.random.uniform((batch_size, 1), 0, 49, tf.int32)
    label_img, feature_img = scheduler(image_batch, time_step)


if __name__ == '__main__':
    test_scheduler()
