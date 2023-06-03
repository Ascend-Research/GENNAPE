import tensorflow.compat.v1 as tf


BN_MOMENTUM = 0.99
BN_EPSILON = 1e-3


def _decode_kernel_size(op):
    k = [int(s) for s in op.split("x")]
    assert all(v % 2 != 0 for v in k), "Only odd kernel sizes supported"
    return k


def _decode_channel_change_ratio(op, code="e"):
    assert op.startswith(code)
    op = op.replace(code, "")
    ratio = int(op)
    return ratio


def _decode_se_ratio(op, delim="_"):
    assert op.startswith(delim)
    op = op.replace(delim, "")
    ratio = float(op) / 100.
    return ratio


def get_op_model(op_name, C_in, stride, scope_name, data_format="channels_last"):
    if op_name.startswith("nor_conv_"):
        k = _decode_kernel_size(op_name.replace("nor_conv_", ""))
        op = AugmentedConv2D(filters=C_in, kernel_size=k, stride=stride, padding="same",
                             activation=tf.keras.layers.ReLU(), name=scope_name,
                             data_format=data_format)
    elif op_name.startswith("avg_pool_"):
        k = _decode_kernel_size(op_name.replace("avg_pool_", ""))
        op = PoolBN(tf.keras.layers.AveragePooling2D, kernel_size=k, stride=stride,
                    padding="same", name=scope_name, data_format=data_format)
    elif op_name == "skip_connect":
        op = Identity(name=scope_name)
    elif op_name == "none":
        op = Zero(name=scope_name)
    else:
        raise ValueError("Unknown op name: {}".format(op_name))
    return op


class AugmentedConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, stride, padding, activation, name,
                 data_format="channels_last", use_bn=True, use_bias=False):
        super(AugmentedConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bn = use_bn
        self.padding = padding
        self.data_format = data_format
        self.activation = activation
        self._name = name
        self.use_bias = use_bias
        if self.data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.stride,
                                           padding=self.padding, data_format=self.data_format,
                                           activation=None, use_bias=self.use_bias,
                                           kernel_initializer="he_normal", name=self._name + "/conv")
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                         momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                         name=self._name + "/bn")
        else:
            self.bn = None

    def call(self, inputs, training=True):
        x = inputs
        if self.activation is not None:
            with tf.name_scope(self._name):
                x = self.activation(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        return x


class PoolBN(tf.keras.layers.Layer):

    def __init__(self, pool_maker, kernel_size, stride, padding, name,
                 data_format="channels_last", use_bn=True):
        super(PoolBN, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.use_bn = use_bn
        self._name = name
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.pool = pool_maker(self.kernel_size, strides=self.stride, padding=self.padding,
                               data_format=self.data_format, name=self._name + "/pool")
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                         momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                         name=self._name + "/bn")
        else:
            self.bn = None

    def call(self, inputs, training=True):
        x = self.pool(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        return x


class Identity(tf.keras.layers.Layer):

    def __init__(self, name="Identity"):
        super(Identity, self).__init__()
        self._name = name

    def call(self, inputs, training=True):
        return tf.identity(inputs, name=self._name)


class Zero(tf.keras.layers.Layer):

    def __init__(self, name="Zero"):
        super(Zero, self).__init__()
        self._name = name

    def call(self, inputs, training=True):
        return tf.identity(inputs, name=self._name) * 0.
