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


def _decode_int_value(op, code="e"):
    assert op.startswith(code)
    op = op.replace(code, "")
    ratio = int(op)
    return ratio


def get_op_model(C_in, C_out, stride, op_name, scope_name,
                 use_bn, use_bias,
                 data_format="channels_last"):
    if op_name.startswith("conv"):
        k = _decode_kernel_size(op_name.replace("conv", ""))
        op = AugmentedConv2D(filters=C_out, kernel_size=k, stride=stride, padding="same",
                             activation=tf.keras.layers.ReLU(max_value=6), name=scope_name,
                             use_bn=use_bn, use_bias=use_bias, data_format=data_format)
    elif op_name.startswith("depthwise"):
        k = _decode_kernel_size(op_name.replace("depthwise", ""))
        op = DepthwiseConv2D(C_in=C_in, C_out=C_out, kernel_size=k, stride=stride, padding="same",
                             use_bn=use_bn, use_bias=use_bias,
                             name=scope_name, data_format=data_format)
    elif op_name.startswith("maxpool"):
        k = _decode_kernel_size(op_name.replace("maxpool", ""))
        op = PoolBN(tf.keras.layers.MaxPooling2D, kernel_size=k, stride=stride,
                    use_bn=use_bn, padding="same", name=scope_name,
                    data_format=data_format)
    elif op_name.startswith("avgpool"):
        k = _decode_kernel_size(op_name.replace("avgpool", ""))
        op = PoolBN(tf.keras.layers.AveragePooling2D, kernel_size=k, stride=stride,
                    use_bn=use_bn, padding="same", name=scope_name,
                    data_format=data_format)
    elif op_name.startswith("inception_s"):
        k = _decode_kernel_size(op_name.replace("inception_s", ""))
        op = InceptionS(C_out, kernel_size=k[0], stride=stride, padding="same",
                        use_bn=use_bn, use_bias=use_bias,
                        data_format=data_format, name=scope_name)
    elif op_name.startswith("inception_p"):
        k = _decode_kernel_size(op_name.replace("inception_p", ""))
        op = InceptionP(C_out, kernel_size=k[0], stride=stride, padding="same",
                        use_bn=use_bn, use_bias=use_bias,
                        data_format=data_format, name=scope_name)
    elif op_name.startswith("double"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k = _decode_kernel_size(conv_op_str.replace("double", ""))
        if len(parts) == 1:
            op = tf.keras.models.Sequential([
                AugmentedConv2D(filters=C_out, kernel_size=k, stride=1, padding="same",
                                activation=tf.keras.layers.ReLU(max_value=6), name=scope_name + "/c1",
                                use_bn=use_bn, use_bias=use_bias,
                                data_format=data_format),
                AugmentedConv2D(filters=C_out, kernel_size=k, stride=stride, padding="same",
                                activation=tf.keras.layers.ReLU(max_value=6), name=scope_name + "/c2",
                                use_bn=use_bn, use_bias=use_bias,
                                data_format=data_format)
            ])
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                op = tf.keras.models.Sequential([
                    AugmentedConv2D(filters=C_hidden, kernel_size=k, stride=stride, padding="same",
                                    activation=tf.keras.layers.ReLU(max_value=6), name=scope_name + "/c1",
                                    use_bn=use_bn, use_bias=use_bias,
                                    data_format=data_format),
                    AugmentedConv2D(filters=C_hidden, kernel_size=k, stride=1, padding="same",
                                    activation=tf.keras.layers.ReLU(max_value=6), name=scope_name + "/c2",
                                    use_bn=use_bn, use_bias=use_bias,
                                    data_format=data_format)
                ])
                op = ChannelResizeWrapper(C_hidden, C_out, op,
                                          use_bn=use_bn,
                                          name=scope_name + "/c_resize")
            else:
                raise ValueError
        else:
            raise ValueError
    elif op_name.startswith("bottleneck"):
        parts = op_name.split("-")
        if len(parts) == 3:
            _, opt1, opt2 = parts
            if opt1.startswith("r"):
                r = _decode_int_value(opt1, "r")
                C_hidden = max(1, C_in // r)
            elif opt1.startswith("e"):
                e = _decode_int_value(opt1, "e")
                C_hidden = int(C_in * e)
            else:
                raise ValueError
            k = _decode_int_value(opt2, "k")
            op = Bottleneck(C_hidden, C_out,
                            use_bn=use_bn,
                            use_bias=use_bias,
                            name=scope_name,
                            kernel_size=k,
                            strides=stride)
        else:
            raise ValueError
    else:
        raise ValueError("Unknown op name: {}".format(op_name))
    return op


class ChannelResizeWrapper(tf.keras.layers.Layer):

    def __init__(self, C_hidden, C_out, op, name,
                 use_bn=True, data_format="channels_last"):
        super(ChannelResizeWrapper, self).__init__()
        self.data_format = data_format
        self._name = name
        self.op = op
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.in_proj = AugmentedConv2D(filters=C_hidden, kernel_size=1, stride=1,
                                       padding="same", data_format=self.data_format,
                                       use_bn=use_bn, use_bias=False, activation=None,
                                       name=self._name + "/in_proj")
        self.out_proj = AugmentedConv2D(filters=C_out, kernel_size=1, stride=1,
                                        padding="same", data_format=self.data_format,
                                        use_bn=use_bn, use_bias=False, activation=None,
                                        name=self._name + "/out_proj")

    def call(self, inputs, training=True):
        x = self.in_proj(inputs, training=training)
        x = self.op(x, training=training)
        x = self.out_proj(x, training=training)
        return x


class AugmentedConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, stride, padding, activation, name,
                 data_format="channels_last", use_bn=True, use_bias=True):
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
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.activation is not None:
            with tf.name_scope(self._name):
                x = self.activation(x)
        return x


class DepthwiseConv2D(tf.keras.layers.Layer):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, name,
                 use_bn=True, use_bias=True, data_format="channels_last"):
        super(DepthwiseConv2D, self).__init__()
        self.filters = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        self.data_format = data_format
        self._name = name
        if self.data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        if C_in != C_out:
            self.proj = AugmentedConv2D(filters=C_out, kernel_size=1, stride=1,
                                        padding="same", data_format=self.data_format,
                                        use_bn=False, use_bias=False, activation=None,
                                        name=self._name + "/in_proj")
        else:
            self.proj = None
        self.depth_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride,
                                                          padding="same", data_format=data_format,
                                                          name=self._name + "/depth_conv", use_bias=use_bias,
                                                          kernel_initializer="he_normal")
        if self.use_bn:
            self.depth_bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                               momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                               name=self._name + "/depth_bn")
        else:
            self.depth_bn = None
        self.relu = tf.keras.layers.ReLU(max_value=6, name=self._name + "/relu6")

    def call(self, inputs, training=True):
        if self.proj is not None:
            inputs = self.proj(inputs, training=training)
        x = self.depth_conv(inputs)
        if self.depth_bn is not None:
            x = self.depth_bn(x, training=training)
        x = self.relu(x)
        return x


class InceptionS(tf.keras.layers.Layer):

    def __init__(self, C_out, kernel_size, stride, padding,
                 use_bn=True, use_bias=True,
                 data_format="channels_last",
                 name="InceptionS"):
        super(InceptionS, self).__init__()
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.data_format = data_format
        self._name = name
        self.conv_1 = AugmentedConv2D(filters=C_out, kernel_size=[1, kernel_size], stride=(1, self.stride),
                                      padding="same", data_format=self.data_format,
                                      use_bn=use_bn, use_bias=use_bias,
                                      activation=tf.keras.layers.ReLU(max_value=6),
                                      name=self._name + "/conv1")
        self.conv_2 = AugmentedConv2D(filters=C_out, kernel_size=[kernel_size, 1], stride=(self.stride, 1),
                                      padding="same", data_format=self.data_format,
                                      use_bn=use_bn, use_bias=use_bias,
                                      activation=tf.keras.layers.ReLU(max_value=6),
                                      name=self._name + "/conv2")

    def call(self, inputs, training=True):
        x = self.conv_1(inputs, training=training)
        output = self.conv_2(x, training=training)
        return output


class InceptionP(tf.keras.layers.Layer):

    def __init__(self, C_out, kernel_size, stride, padding,
                 use_bn=True, use_bias=True,
                 data_format="channels_last",
                 name="InceptionP"):
        super(InceptionP, self).__init__()
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.data_format = data_format
        self._name = name
        C1 = C_out // 2
        if self.data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        if self.stride == 1:
            self.conv_1 = AugmentedConv2D(filters=C1, kernel_size=[1, kernel_size], stride=1,
                                          padding="same", data_format=self.data_format,
                                          use_bn=use_bn, use_bias=use_bias,
                                          activation=None, name=self._name + "/conv1")
            self.conv_2 = AugmentedConv2D(filters=C_out - C1, kernel_size=[kernel_size, 1], stride=1,
                                          padding="same", data_format=self.data_format,
                                          use_bn=use_bn, use_bias=use_bias,
                                          activation=None, name=self._name + "/conv2")
        else:
            self.conv_1 = tf.keras.models.Sequential([
                AugmentedConv2D(filters=C1, kernel_size=[1, kernel_size], stride=(1, self.stride),
                                padding="same", data_format=self.data_format,
                                use_bn=use_bn, use_bias=use_bias,
                                activation=None, name=self._name + "/conv1/p1"),
                AugmentedConv2D(filters=C1, kernel_size=[kernel_size, 1], stride=(self.stride, 1),
                                padding="same", data_format=self.data_format,
                                use_bn=use_bn, use_bias=use_bias,
                                activation=None, name=self._name + "/conv1/p2"),
            ])
            self.conv_2 =tf.keras.models.Sequential([
                AugmentedConv2D(filters=C_out - C1, kernel_size=[kernel_size, 1], stride=(self.stride, 1),
                                padding="same", data_format=self.data_format,
                                use_bn=use_bn, use_bias=use_bias,
                                activation=None, name=self._name + "/conv2/p1"),
                AugmentedConv2D(filters=C_out - C1, kernel_size=[1, kernel_size], stride=(1, self.stride),
                                padding="same", data_format=self.data_format,
                                use_bn=use_bn, use_bias=use_bias,
                                activation=None, name=self._name + "/conv2/p2"),
            ])
        self.activ = tf.keras.layers.ReLU(max_value=6, name=self._name + "/relu")

    def call(self, inputs, training=True):
        x1 = self.conv_1(inputs, training=training)
        x2 = self.conv_2(inputs, training=training)
        output = tf.concat([x1, x2], self.channel_axis,
                           name=self._name + "/concat")
        return self.activ(output)


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


class Bottleneck(tf.keras.layers.Layer):

    def __init__(self, C_hidden, C_out, kernel_size, strides,
                 use_bn=True, use_bias=True,
                 padding="same", name="Bottleneck",
                 data_format="channels_last"):
        super(Bottleneck, self).__init__()
        self._name = name
        self.squeeze_conv = AugmentedConv2D(filters=C_hidden, kernel_size=kernel_size, stride=strides,
                                            padding=padding, data_format=data_format,
                                            use_bn=use_bn, use_bias=use_bias, activation=None,
                                            name=self._name + "/s_conv")
        self.excite_conv = AugmentedConv2D(filters=C_out, kernel_size=kernel_size, stride=1,
                                           padding="same", data_format=data_format,
                                           use_bn=use_bn, use_bias=use_bias, activation=None,
                                           name=self._name + "/e_conv")

    def call(self, inputs, training=True):
        x = self.squeeze_conv(inputs, training=training)
        x = self.excite_conv(x, training=training)
        return x
