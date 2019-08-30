import tensorflow as tf
from tf.keras import backend, layers, models


class MobileNetV3:
    def __init__(self, shape):
        self.shape = shape

    def build(self, shape):
        inputs = layers.Input(shape=self.shape)

        x = layers.Conv2D(filters, kernel, padding="same", strides=strides)(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = self._activation(x, "HS")

        x = self._bneck(x, 16, (3, 3), expansion=16, strides=1, squeeze=False, nl="RE")
        x = self._bneck(x, 24, (3, 3), expansion=64, strides=2, squeeze=False, nl="RE")
        x = self._bneck(x, 24, (3, 3), expansion=72, strides=1, squeeze=False, nl="RE")
        x = self._bneck(x, 40, (5, 5), expansion=72, strides=2, squeeze=True, nl="RE")
        x = self._bneck(x, 40, (5, 5), expansion=120, strides=1, squeeze=True, nl="RE")
        x = self._bneck(x, 40, (5, 5), expansion=120, strides=1, squeeze=True, nl="RE")
        x = self._bneck(x, 80, (3, 3), expansion=240, strides=2, squeeze=False, nl="HS")
        x = self._bneck(x, 80, (3, 3), expansion=200, strides=1, squeeze=False, nl="HS")
        x = self._bneck(x, 80, (3, 3), expansion=184, strides=1, squeeze=False, nl="HS")
        x = self._bneck(x, 80, (3, 3), expansion=184, strides=1, squeeze=False, nl="HS")
        x = self._bneck(x, 112, (3, 3), expansion=480, strides=1, squeeze=True, nl="HS")
        x = self._bneck(x, 112, (3, 3), expansion=672, strides=1, squeeze=True, nl="HS")
        x = self._bneck(x, 160, (5, 5), expansion=672, strides=2, squeeze=True, nl="HS")
        x = self._bneck(x, 160, (5, 5), expansion=960, strides=1, squeeze=True, nl="HS")
        x = self._bneck(x, 160, (5, 5), expansion=960, strides=1, squeeze=True, nl="HS")

        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl="HS")
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, 960))(x)

        x = layers.Conv2D(1280, (1, 1), padding="same")(x)
        x = self._activation(x, "HS")

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding="same", activation="softmax")(x)
            x = Reshape((self.n_class,))(x)

        model = models.Model(inputs, x)

        return model

    def train():
        raise NotImplementedError

    def eval():
        raise NotImplementedError

    def _bneck(self, x, filters, kernel, expansion, strides, squeeze, at):
        x_copy = x

        input_shape = backend.int_shape(x)
        tchannel = int(expansion)
        cchannel = int(filters)

        r = strides == 1 and input_shape[3] == filters

        x = layers.Conv2D(filters, kernel, padding="same", strides=strides)(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = self._activation(x, at)
        x = layers.DepthwiseConv2D(
            kernel, strides=(strides, strides), depth_multiplier=1, padding="same"
        )(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = self._return_activation(x, at)

        if squeeze:
            x = self._squeeze()

        x = layers.Conv2D(cchannel, (1, 1), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)

        if r:
            x = layers.Add()([x, x_copy])

        return x

    def _squeeze(self, x):
        x_copy = x
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(3, activation="relu")(x)
        x = layers.Dense(3, activation="hard_sigmoid")(x)
        x = layers.Reshape((1, 1, 3))(x)
        x = layers.Multiply()([x_copy, x])

        return x

    def _activation(self, x, at):
        if at == "RE":
            # ReLU6
            x = backend.relu(x, max_value=6)
        else:
            # Hard swish
            x = x * backend.relu(x, max_value=6) / 6
