from tensorflow import keras


class AbstractConfig(keras.layers.Layer):
    def get_config(self):
        return super().get_config()
