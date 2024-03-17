import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras


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
