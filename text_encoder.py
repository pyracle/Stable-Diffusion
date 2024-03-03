import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # needed to load the models from tensorflow hub
from tensorflow import keras
from base_config import AbstractConfig


class TextEncoder(keras.Model,
                  AbstractConfig):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = hub.KerasLayer(
            'https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/'
            'variations/en-uncased-preprocess/versions/3'
        )
        self.encoder = hub.KerasLayer(
            'https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/'
            'variations/bert-en-uncased-l-2-h-768-a-12/versions/2',
            trainable=True
        )
        self.conv = keras.layers.Conv1D(512, 3,  padding='same', activation='relu')
        self.norm = keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.preprocessor(inputs)
        x = self.encoder(x, training=training)['sequence_output']
        x = self.conv(x, training=training)

        assert x.get_shape() == (inputs.shape[0], 128, 512)
        return self.norm(x)


def test_text_encoder():
    model = TextEncoder()
    model(tf.constant(['this is a test']))
    print(model.summary())


if __name__ == '__main__':
    test_text_encoder()
