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
    
    
class FeedForward(AbstractConfig):
    def __init__(self,
                 d_model: int,
                 dff: int,
                 dropout_rate: float):
        super().__init__()
        self.seq = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = self.seq(inputs, training=training)
        x += inputs
        return self.layer_norm(x)


class VisionTransformerEncoderBlock(AbstractConfig):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 dropout_rate=0.1,
                 **kwargs):
        super(VisionTransformerEncoderBlock, self).__init__(**kwargs)
        self.self_attention = Attention(
            num_heads=num_heads,
            key_dim=d_model,
        )
        self.ffn = FeedForward(
            d_model,
            dff,
            dropout_rate
        )

    def call(self, inputs, training=False):
        x = self.self_attention(inputs, inputs, training=training)
        self.last_attn_scores = self.self_attention.last_attn_scores
        return self.ffn(x)


class VisionTransformerEncoder(AbstractConfig):
    def __init__(self,
                 image_size: int,
                 channels: int,
                 patch_size: int,
                 d_model: int,
                 num_heads: int,
                 num_blocks: int,
                 dff: int,
                 dropout_rate=0.1,
                 **kwargs):
        super(VisionTransformerEncoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.patch_dim = channels * patch_size ** 2
        self.dense_patch = keras.layers.Dense(d_model)
        num_patches = (image_size // patch_size) ** 2
        self.pos_emb = self.add_weight(
            'pos_emb',
            (1, num_patches + 1, d_model),
            dtype=tf.float32
        )
        self.class_emb = self.add_weight('class_emb', (1, 1, d_model))
        self.encoder_blocks = [
            VisionTransformerEncoderBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_blocks)
        ]

    @tf.function
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        return tf.reshape(patches, [batch_size, -1, self.patch_dim])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        x = self.extract_patches(inputs)
        x = self.dense_patch(x)
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x += self.pos_emb

        for block_idx in range(self.num_blocks):
            x = self.encoder_blocks[block_idx](x)

        self.last_attn_scores = self.encoder_blocks[-1].last_attn_scores
        return x
    
    
class MLPHead(AbstractConfig):
    def __init__(self,
                 dff: int,
                 d_model: int,
                 n_classes: int,
                 dropout_rate: float,
                 **kwargs):
        super(MLPHead, self).__init__(**kwargs)
        self.seq = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(n_classes)
        ])
        
    def call(self, inputs, training=False):
        return self.seq(inputs, training=training)


class VisionTransformer(keras.Model,
                        AbstractConfig):
    def __init__(self,
                 image_size: int,
                 channels: int,
                 patch_size: int,
                 d_model: int,
                 num_heads: int,
                 num_blocks: int,
                 num_classes: int,
                 dff: int,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.encoder = VisionTransformerEncoder(
            image_size,
            channels,
            patch_size,
            d_model,
            num_heads,
            num_blocks,
            dff,
            dropout_rate
        )
        self.mlp_head = MLPHead(
            dff,
            d_model,
            num_classes,
            dropout_rate
        )
    
    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        return self.mlp_head(x, training=training)


class UNetBlock(AbstractConfig):
    def __init__(self,
                 **kwargs):
        super(UNetBlock, self).__init__(**kwargs)
        self.conv = keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.dense_time = keras.layers.Dense(256)
        self.reshape_time = keras.layers.Reshape((1, 1, 256))
        self.conv_output = keras.layers.Conv2D(256, 5, padding='same')
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
                 dff,
                 d_model: int,
                 num_attention_heads: int,
                 dropout_rate: float = .1,
                 repetitions: int = 50,
                 num_vit_encoder_blocks: int = 8,
                 **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.dense_time = keras.layers.Dense(256)
        self.unet_blocks = [UNetBlock() for _ in range(8)]
        self.dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in range(8)]
        self.attention_layers = [Attention(
            num_heads=num_attention_heads,
            key_dim=d_model
        ) for _ in range(8)]
        self.max_pool = [keras.layers.MaxPooling2D() for _ in range(3)]
        self.vit = VisionTransformer(
            image_size=8,
            channels=256,
            patch_size=2,
            d_model=d_model,
            num_heads=num_attention_heads,
            num_blocks=num_vit_encoder_blocks,
            num_classes=1024,
            dff=dff,
            dropout_rate=dropout_rate
        )
        self.reshape = keras.layers.Reshape((8, 8, 16))
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

        assert x_8.get_shape()[1:] == (8, 8, 256)
        
        x = self.vit(x_8)

        assert x.get_shape()[1:] == (1024)

        x = self.reshape(x)
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
        epoch_addition =  '/saved-model-{epoch:'f'{len(str(epochs))}''.0f}'
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
            image = self.model.unet((image, text, time_step))
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
    unet = UNet(dff=64,
                d_model=256,
                num_attention_heads=8)
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
    unet = UNet(dff=64,
                d_model=256,
                num_attention_heads=8,
                dropout_rate=.1,
                repetitions=unet_repetitions)
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
