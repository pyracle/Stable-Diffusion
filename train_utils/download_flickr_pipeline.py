import pathlib
import collections
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras


class DataLoader(keras.layers.Layer):
    def __init__(self,
                 batch_size: int = 32,
                 data_dir: str = 'flickr8k'):
        super().__init__()
        self.batch_size = batch_size
        self.flickr8k(path=data_dir)

    def call(self, mode: str = 'both'):
        return (
            self.preprocess_dataset(self.train_ds, mode),
            self.preprocess_dataset(self.test_ds, mode)
        )

    def preprocess_dataset(self, ds: tf.data.Dataset, mode: str):
        autotune = tf.data.AUTOTUNE
        if mode == 'text':
            ds = ds.map(lambda img, text: self.get_first_caption(text))
        elif mode == 'image':
            ds = ds.map(lambda img, text: self.load_image(img))
        elif mode == 'both':
            ds = ds.map(lambda img, text: (
                self.load_image(img),
                self.get_first_caption(text),
            ))
        else:
            raise ValueError(f'Mode has to be either image, text or both')

        ds = ds.batch(self.batch_size, drop_remainder=True, num_parallel_calls=autotune)
        ds = ds.shuffle(10 * self.batch_size)
        ds = ds.cache().prefetch(autotune)
        return ds
    
    @staticmethod
    def get_first_caption(text):
        return text[0]

    @staticmethod
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, 3)
        img = tf.image.resize(img, (256, 256))
        return tf.cast(img, tf.float32) / 127.5 - 1

    @classmethod
    def flickr8k(cls, path='flickr8k'):
        """
        published at tensorflow.org/text/tutorials/image_captioning?hl=en#optional_data_handling
        """
        
        path = pathlib.Path(path)

        if len(list(path.rglob('*'))) < 16197:
            tf.keras.utils.get_file(
                origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                cache_dir='.',
                cache_subdir=path,
                extract=True
            )
            tf.keras.utils.get_file(
                origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                cache_dir='.',
                cache_subdir=path,
                extract=True
            )

        captions = (path / "Flickr8k.token.txt").read_text().splitlines()
        captions = (line.split('\t') for line in captions)
        captions = ((filename.split('#')[0], caption) for (filename, caption) in captions)

        captions_dict = collections.defaultdict(list)
        for filename, caption in captions:
            captions_dict[filename].append(caption)

        train_files = (path / 'Flickr_8k.trainImages.txt').read_text().splitlines()
        train_captions = [
            (str(path / 'Flicker8k_Dataset' / filename), captions_dict[filename])
            for filename in train_files
        ]

        test_files = (path / 'Flickr_8k.testImages.txt').read_text().splitlines()
        test_captions = [
            (str(path / 'Flicker8k_Dataset' / filename), captions_dict[filename])
            for filename in test_files
        ]

        cls.train_ds = tf.data.experimental.from_list(train_captions)
        cls.test_ds = tf.data.experimental.from_list(test_captions)


def test_data_loader():
    data_loader = DataLoader(data_dir='data/flickr8k')
    train_ds, test_ds = data_loader('image')
    print(test_ds.as_numpy_iterator().next())


if __name__ == "__main__":
    test_data_loader()
