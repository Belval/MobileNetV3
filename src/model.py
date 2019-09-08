import tensorflow as tf
from tensorflow.keras import backend, layers, models


class MobileNetV3LiteRASPP:
    def __init__(self, shape, n_class):
        self.shape = shape
        self.n_class = n_class

    def build(self):
        # See https://arxiv.org/pdf/1905.02244v4.pdf p.5 Table 1
        inputs = layers.Input(shape=self.shape)

        print(backend.int_shape(inputs))

        x = layers.Conv2D(16, (3, 3), padding="same", strides=(2, 2))(inputs)
        x = layers.BatchNormalization(axis=-1)(x)
        x = self._activation(x, "HS")

        print(backend.int_shape(x))


        # Bottleneck blocks
        
        # 1/2
        x, _, _, _ = self._bneck(x, 16, (3, 3), expansion=16, strides=1, squeeze=False, at="RE")
        
        print(backend.int_shape(x))

        # 1/4
        x, _, _, _ = self._bneck(x, 24, (3, 3), expansion=64, strides=2, squeeze=False, at="RE")
        x, _, _, _ = self._bneck(x, 24, (3, 3), expansion=72, strides=1, squeeze=False, at="RE")

        print(backend.int_shape(x))
        
        # 1/8
        x, _, _, _ = self._bneck(x, 40, (5, 5), expansion=72, strides=2, squeeze=True, at="RE")
        x, _, _, _ = self._bneck(x, 40, (5, 5), expansion=120, strides=1, squeeze=True, at="RE")
        x_8, _, _, _ = self._bneck(x, 40, (5, 5), expansion=120, strides=1, squeeze=True, at="RE")
        
        print(backend.int_shape(x_8))

        # 1/16
        x, _, _, _ = self._bneck(x_8, 80, (3, 3), expansion=240, strides=2, squeeze=False, at="HS")
        x, _, _, _ = self._bneck(x, 80, (3, 3), expansion=200, strides=1, squeeze=False, at="HS")
        x, _, _, _ = self._bneck(x, 80, (3, 3), expansion=184, strides=1, squeeze=False, at="HS")
        x, _, _, _ = self._bneck(x, 80, (3, 3), expansion=184, strides=1, squeeze=False, at="HS")
        x, _, _, _ = self._bneck(x, 112, (3, 3), expansion=480, strides=1, squeeze=True, at="HS")
        x_16, _, _, _ = self._bneck(x, 112, (3, 3), expansion=672, strides=1, squeeze=True, at="HS")
        
        print(backend.int_shape(x_16))

        # 1/32
        # 13th bottleneck block (C4) https://arxiv.org/pdf/1905.02244v4.pdf p.7
        #x, _, _, _ = self._bneck(x_18, 160, (5, 5), expansion=672, strides=2, squeeze=True, at="HS")
        #x, _, _, _ = self._bneck(x, 160, (5, 5), expansion=960, strides=1, squeeze=True, at="HS")
        #x, _, _, _ = self._bneck(x, 160, (5, 5), expansion=960, strides=1, squeeze=True, at="HS")
        #
        #print(backend.int_shape(x))

        # Layer immediatly before pooling (C5) https://arxiv.org/pdf/1905.02244v4.pdf p.7
        #x = layers.Conv2D(960, (1, 1), strides=(1, 1), padding="same")(x)
        #x = layers.BatchNormalization(axis=-1)(x)
        #x = self._activation(x, at="HS")

        #print(backend.int_shape(x))

        ## Pooling layer
        #x = layers.GlobalAveragePooling2D()(x)
        #x = layers.Reshape((1, 1, 960))(x)
        #
        #x = layers.Conv2D(1280, (1, 1), padding="same")(x)
        #x = self._activation(x, "HS")
        #
        x = self._segmentation_head(x_16, x_8)

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

        # Expansion convolution
        exp_x = layers.Conv2D(tchannel, (1, 1), padding="same", strides=(1, 1))(x)
        exp_x = layers.BatchNormalization(axis=-1)(exp_x)
        exp_x = self._activation(exp_x, at)

        # Depthwise convolution
        dep_x = layers.DepthwiseConv2D(
            kernel, strides=(strides, strides), depth_multiplier=1, padding="same"
        )(exp_x)
        dep_x = layers.BatchNormalization(axis=-1)(dep_x)
        dep_x = self._activation(dep_x, at)

        # Squeeze
        if squeeze:
            dep_x = self._squeeze(dep_x)

        # Projection convolution
        pro_x = layers.Conv2D(cchannel, (1, 1), strides=(1, 1), padding="same")(dep_x)
        pro_x = layers.BatchNormalization(axis=-1)(pro_x)

        x = pro_x

        if r:
            print('Add')
            x = layers.Add()([pro_x, x_copy])

        return x, exp_x, dep_x, pro_x

    def _segmentation_head(self, x_16, x_8):
        x_copy = x_16
        # First branch
        x_b1 = layers.Conv2D(128, (1, 1), strides=(1, 1), padding="same")(x_16)

        # This is the size we want for the other branch
        s = x_b1.shape

        x_b1 = layers.BatchNormalization(axis=-1)(x_b1)
        x_b1 = self._activation(x_b1, at="RE")

        # Second branch
        x_b2 = layers.AveragePooling2D(pool_size=(25, 25), strides=(8, 10))(x_16)
        x_b2 = layers.Conv2D(128, (1, 1))(x_b2)
        x_b2 = layers.Activation('sigmoid')(x_b2)
        x_b2 = layers.UpSampling2D(size=(int(s[1]), int(s[2])), interpolation="bilinear")(x_b2)

        # Merging branches 1 and 2
        x = layers.Multiply()([x_b1, x_b2])
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(self.n_class, (1, 1))(x)

        x_b3 = layers.Conv2D(self.n_class, (1, 1))(x_8)

        # Merging merge 1 and branche 3
        x = layers.Add()([x, x_b3])
        x = layers.Activation('softmax')(x)

        return x

    def _squeeze(self, x):
        x_copy = x
        channel = backend.int_shape(x)[-1]
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(channel, activation="relu")(x)
        x = layers.Dense(channel, activation="hard_sigmoid")(x)
        x = layers.Reshape((1, 1, channel))(x)
        x = layers.Multiply()([x_copy, x])

        return x

    def _activation(self, x, at):
        if at == "RE":
            # ReLU6
            x = backend.relu(x, max_value=6)
        else:
            # Hard swish
            x = x * backend.relu(x, max_value=6) / 6

        return x