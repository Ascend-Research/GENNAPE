import tensorflow.compat.v1 as tf


BN_MOMENTUM = 0.9
BN_EPSILON = 1e-3


def _decode_kernel_size(op):
    k = [int(s) for s in op.split("x")]
    assert all(v % 2 != 0 for v in k), "Only odd kernel sizes supported"
    return k


def _decode_int_value(op, code="e"):
    assert op.startswith(code)
    op = op.replace(code, "")
    ratio = int(op)
    return ratio


def get_two_path_op_model(C_in, C_out, stride, op_name, scope_name,
                          data_format="channels_last"):
    if op_name.startswith("conv"):
        k = _decode_kernel_size(op_name.replace("conv", ""))
        op = AugmentedConv2D(filters=C_out, kernel_size=k, stride=stride, padding="same",
                             activation=tf.keras.layers.ReLU(max_value=6), name=scope_name,
                             data_format=data_format)
    elif op_name.startswith("maxpool"):
        k = _decode_kernel_size(op_name.replace("maxpool", ""))
        op = PoolBN(tf.keras.layers.MaxPooling2D, kernel_size=k, stride=stride,
                    padding="same", name=scope_name, data_format=data_format)
    elif op_name.startswith("avgpool"):
        k = _decode_kernel_size(op_name.replace("avgpool", ""))
        op = PoolBN(tf.keras.layers.AveragePooling2D, kernel_size=k, stride=stride,
                    padding="same", name=scope_name, data_format=data_format)
    elif op_name.startswith("inception_s"):
        k = _decode_kernel_size(op_name.replace("inception_s", ""))
        op = InceptionS(C_out, kernel_size=k[0], stride=stride, padding="same",
                        data_format=data_format, name=scope_name)
    elif op_name.startswith("double"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k = _decode_kernel_size(conv_op_str.replace("double", ""))
        if len(parts) == 1:
            op = tf.keras.models.Sequential([
                AugmentedConv2D(filters=C_in, kernel_size=k, stride=stride, padding="same",
                                activation=tf.keras.layers.ReLU(max_value=6), name=scope_name + "/c1",
                                data_format=data_format),
                AugmentedConv2D(filters=C_out, kernel_size=k, stride=1, padding="same",
                                activation=tf.keras.layers.ReLU(max_value=6), name=scope_name + "/c2",
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
                                    data_format=data_format),
                    AugmentedConv2D(filters=C_hidden, kernel_size=k, stride=1, padding="same",
                                    activation=tf.keras.layers.ReLU(max_value=6), name=scope_name + "/c2",
                                    data_format=data_format)
                ])
                op = ChannelResizeWrapper(C_hidden, C_out, op,
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
                            name=scope_name,
                            kernel_size=k,
                            strides=stride)
        else:
            raise ValueError
    else:
        raise ValueError("Unknown op name: {}".format(op_name))
    return op


def get_hiaml_block_op_model(C_in, C_out, stride, op_name, scope_name,
                             data_format="channels_last"):
    if op_name == "hiaml_a22":
        op = BlockA22(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_z":
        op = BlockZ(C_in, C_out,
                    b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_j30":
        op = BlockJ30(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_b23":
        op = BlockB23(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_g30":
        op = BlockG30(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_r30":
        op = BlockR30(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_l20":
        op = BlockL20(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_t20":
        op = BlockT20(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_t31":
        op = BlockT31(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_src0":
        op = BlockSrc0(C_in, C_out,
                       b_stride=stride, name=scope_name,
                       data_format=data_format)

    elif op_name == "hiaml_src1":
        op = BlockSrc1(C_in, C_out,
                       b_stride=stride, name=scope_name,
                       data_format=data_format)

    elif op_name == "hiaml_src3":
        op = BlockSrc3(C_in, C_out,
                      b_stride=stride, name=scope_name,
                      data_format=data_format)

    elif op_name == "hiaml_src7":
        op = BlockSrc7(C_in, C_out,
                       b_stride=stride, name=scope_name,
                       data_format=data_format)

    elif op_name == "hiaml_src8":
        op = BlockSrc8(C_in, C_out,
                       b_stride=stride, name=scope_name,
                       data_format=data_format)

    else:
        raise ValueError("Unknown op name: {}".format(op_name))
    return op


class ChannelResizeWrapper(tf.keras.layers.Layer):

    def __init__(self, C_hidden, C_out, op, name,
                 data_format="channels_last"):
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
                                       use_bn=True, use_bias=False, activation=None,
                                       name=self._name + "/in_proj")
        self.out_proj = AugmentedConv2D(filters=C_out, kernel_size=1, stride=1,
                                        padding="same", data_format=self.data_format,
                                        use_bn=True, use_bias=False, activation=None,
                                        name=self._name + "/out_proj")

    def call(self, inputs, training=True):
        x = self.in_proj(inputs, training=training)
        x = self.op(x, training=training)
        x = self.out_proj(x, training=training)
        return x


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
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.activation is not None:
            with tf.name_scope(self._name):
                x = self.activation(x)
        return x


class DepthwiseConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, stride, padding, name,
                 data_format="channels_last"):
        super(DepthwiseConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self._name = name
        if self.data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.depth_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride,
                                                          padding="same", data_format=data_format,
                                                          name=self._name + "/depth_conv", use_bias=False,
                                                          kernel_initializer="he_normal")
        self.depth_bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                           momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                           name=self._name + "/depth_bn")
        self.relu = tf.keras.layers.ReLU(max_value=6, name=self._name + "/relu6")

    def call(self, inputs, training=True):
        x = self.depth_conv(inputs)
        x = self.depth_bn(x, training=training)
        x = self.relu(x)
        return x


class InceptionS(tf.keras.layers.Layer):

    def __init__(self, C_out, kernel_size, stride, padding,
                 data_format="channels_last", name="InceptionS"):
        super(InceptionS, self).__init__()
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.data_format = data_format
        self._name = name
        self.conv_1 = AugmentedConv2D(filters=C_out, kernel_size=[1, kernel_size], stride=self.stride,
                                      padding="same", data_format=self.data_format,
                                      use_bn=True, use_bias=False,
                                      activation=tf.keras.layers.ReLU(max_value=6),
                                      name=self._name + "/conv1")
        self.conv_2 = AugmentedConv2D(filters=C_out, kernel_size=[kernel_size, 1], stride=1,
                                      padding="same", data_format=self.data_format,
                                      use_bn=True, use_bias=False,
                                      activation=tf.keras.layers.ReLU(max_value=6),
                                      name=self._name + "/conv2")

    def call(self, inputs, training=True):
        x = self.conv_1(inputs, training=training)
        output = self.conv_2(x, training=training)
        return output


class SqueezeExcite(tf.keras.layers.Layer):

    def __init__(self, C_in, C_squeeze, name="SqueezeExcite", data_format="channels_last"):
        super(SqueezeExcite, self).__init__()
        self._name = name
        self.squeeze_conv = tf.keras.layers.Conv2D(filters=C_squeeze, kernel_size=1, strides=1,
                                                   padding="same", data_format=data_format,
                                                   activation=None, use_bias=True,
                                                   kernel_initializer="he_normal",
                                                   name=self._name + "/s_conv")
        self.relu = tf.keras.layers.ReLU(max_value=6, name=self._name + "/relu6")
        self.excite_conv = tf.keras.layers.Conv2D(filters=C_in, kernel_size=1, strides=1,
                                                  padding="same", data_format=data_format,
                                                  activation=None, use_bias=True,
                                                  kernel_initializer="he_normal",
                                                  name=self._name + "/e_conv")
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.spatial_dims = [2, 3]
        else:
            self.channel_axis = 3
            self.spatial_dims = [1, 2]

    def call(self, inputs, training=True):
        with tf.name_scope(self._name):
            x = tf.reduce_mean(inputs, self.spatial_dims, keepdims=True)
        x = self.squeeze_conv(x)
        x = self.relu(x)
        x = self.excite_conv(x)
        with tf.name_scope(self._name):
            return tf.nn.sigmoid(x) * inputs


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
                 padding="same", name="Bottleneck",
                 data_format="channels_last"):
        super(Bottleneck, self).__init__()
        self._name = name
        self.squeeze_conv = AugmentedConv2D(filters=C_hidden, kernel_size=kernel_size, stride=strides,
                                            padding=padding, data_format=data_format,
                                            use_bn=True, use_bias=False, activation=None,
                                            name=self._name + "/s_conv")
        self.excite_conv = AugmentedConv2D(filters=C_out, kernel_size=kernel_size, stride=1,
                                           padding="same", data_format=data_format,
                                           use_bn=True, use_bias=False, activation=None,
                                           name=self._name + "/e_conv")

    def call(self, inputs, training=True):
        x = self.squeeze_conv(inputs, training=training)
        x = self.excite_conv(x, training=training)
        return x


class BlockA22(tf.keras.layers.Layer):
    """
    BlockState[5]:input,conv3x3,conv1x1,conv1x1,output|Edges: [(0, 1), (1, 2), (1, 4), (2, 3), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockA22", data_format="channels_last"):
        super(BlockA22, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv33 = AugmentedConv2D(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv33_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.conv11_2 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                        activation=tf.keras.layers.ReLU(), name=self._name + "/conv11_2",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.conv11_3 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                        activation=None, name=self._name + "/conv11_3",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        out_1 = self.conv33(inputs, training=training)
        out_2 = self.conv11_2(out_1, training=training)
        out_3 = self.conv11_3(out_2, training=training)
        output = self.relu(out_1 + out_3)
        return output


class BlockZ(tf.keras.layers.Layer):
    """
    BlockState[4]:input,conv3x3,inception_s,output|Edges: [(0, 1), (1, 2), (1, 3), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockZ", data_format="channels_last"):
        super(BlockZ, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv33 = AugmentedConv2D(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv33_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.inception_s = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                      data_format=data_format, name=self._name + "/inception_s_2")
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        out_1 = self.conv33(inputs, training=training)
        out_2 = self.inception_s(out_1, training=training)
        output = self.relu(out_1 + out_2)
        return output


class BlockB23(tf.keras.layers.Layer):
    """
    BlockState[4]:input,inception_s,inception_s,output|Edges: [(0, 1), (0, 3), (1, 2), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockB23", data_format="channels_last"):
        super(BlockB23, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.inception_s_1 = InceptionS(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_1")
        self.inception_s_2 = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_2")
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_3 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                                activation=None, name=self._name + "/proj_3",
                                                data_format=data_format, use_bias=False, use_bn=True)
        else:
            self.input_proj_3 = None
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        if self.input_proj_3 is not None:
            in_3 = self.input_proj_3(inputs, training=training)
        else:
            in_3 = inputs
        out_1 = self.inception_s_1(inputs, training=training)
        out_2 = self.inception_s_2(out_1, training=training)
        output = self.relu(in_3 + out_2)
        return output


class BlockJ30(tf.keras.layers.Layer):
    """
    BlockState[4]:input,conv3x3,inception_s,output|Edges: [(0, 1), (0, 3), (1, 2), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockJ30", data_format="channels_last"):
        super(BlockJ30, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv33 = AugmentedConv2D(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv33_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.inception_s = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                      data_format=data_format, name=self._name + "/inception_s_2")
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_3 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                                activation=None, name=self._name + "/proj_3",
                                                data_format=data_format, use_bias=False, use_bn=True)
        else:
            self.input_proj_3 = None
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        if self.input_proj_3 is not None:
            in_3 = self.input_proj_3(inputs, training=training)
        else:
            in_3 = inputs
        out_1 = self.conv33(inputs, training=training)
        out_2 = self.inception_s(out_1, training=training)
        output = self.relu(in_3 + out_2)
        return output


class BlockG30(tf.keras.layers.Layer):
    """
    BlockState[5]:input,inception_s,inception_s,conv1x1,output|
    Edges: [(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockG30", data_format="channels_last"):
        super(BlockG30, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.inception_s_1 = InceptionS(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_1")
        self.inception_s_2 = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_2")
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                      activation=None, name=self._name + "/conv11_3",
                                      data_format=data_format, use_bias=False, use_bn=True)
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_2 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                                activation=None, name=self._name + "/proj_2",
                                                data_format=data_format, use_bias=False, use_bn=True)
        else:
            self.input_proj_2 = None
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        if self.input_proj_2 is not None:
            in_2 = self.input_proj_2(inputs, training=training)
        else:
            in_2 = inputs
        out_1 = self.inception_s_1(inputs, training=training)
        out_2 = self.inception_s_2(in_2 + out_1, training=training)
        out_3 = self.conv11(out_2, training=training)
        output = self.relu(out_1 + out_3)
        return output


class BlockR30(tf.keras.layers.Layer):
    """
    BlockState[5]:input,inception_s,conv1x1,inception_s,output|
    Edges: [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockR30", data_format="channels_last"):
        super(BlockR30, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.inception_s_1 = InceptionS(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_1")
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                      activation=None, name=self._name + "/conv11_2",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.inception_s_3 = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_3")
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_2 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                                activation=None, name=self._name + "/proj_2",
                                                data_format=data_format, use_bias=False, use_bn=True)
            self.input_proj_4 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                                activation=None, name=self._name + "/proj_4",
                                                data_format=data_format, use_bias=False, use_bn=True)
        else:
            self.input_proj_2 = None
            self.input_proj_4 = None
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        if self.input_proj_2 is not None:
            in_2 = self.input_proj_2(inputs, training=training)
            in_4 = self.input_proj_4(inputs, training=training)
        else:
            in_2 = inputs
            in_4 = inputs
        out_1 = self.inception_s_1(inputs, training=training)
        out_2 = self.conv11(in_2 + out_1, training=training)
        out_3 = self.inception_s_3(out_1, training=training)
        output = self.relu(in_4 + out_2 + out_3)
        return output


class BlockL20(tf.keras.layers.Layer):
    """
    BlockState[4]:input,conv3x3,conv1x1,output|Edges: [(0, 1), (1, 2), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockL20", data_format="channels_last"):
        super(BlockL20, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv33 = AugmentedConv2D(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv33_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                      activation=None, name=self._name + "/conv11_2",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        out_1 = self.conv33(inputs, training=training)
        out_2 = self.conv11(out_1, training=training)
        output = self.relu(out_2)
        return output


class BlockT20(tf.keras.layers.Layer):
    """
    BlockState[4]:input,conv1x1,inception_s,output|Edges: [(0, 1), (0, 2), (1, 3), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockT20", data_format="channels_last"):
        super(BlockT20, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                      activation=None, name=self._name + "/conv11_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.inception_s = InceptionS(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                      data_format=data_format, name=self._name + "/inception_s_2")
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        out_1 = self.conv11(inputs, training=training)
        out_2 = self.inception_s(inputs, training=training)
        output = self.relu(out_1 + out_2)
        return output


class BlockT31(tf.keras.layers.Layer):
    """
    BlockState[5]:input,inception_s,conv1x1,inception_s,output|
    Edges: [(0, 1), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockT31", data_format="channels_last"):
        super(BlockT31, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.inception_s_1 = InceptionS(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_1")
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv11_2",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.inception_s_3 = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_3")
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_3 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                                activation=None, name=self._name + "/proj_3",
                                                data_format=data_format, use_bias=False, use_bn=True)
            self.input_proj_4 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                                activation=None, name=self._name + "/proj_4",
                                                data_format=data_format, use_bias=False, use_bn=True)
        else:
            self.input_proj_3 = None
            self.input_proj_4 = None
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        if self.input_proj_3 is not None:
            in_3 = self.input_proj_3(inputs, training=training)
            in_4 = self.input_proj_4(inputs, training=training)
        else:
            in_3 = inputs
            in_4 = inputs
        out_1 = self.inception_s_1(inputs, training=training)
        out_2 = self.conv11(out_1, training=training)
        out_3 = self.inception_s_3(in_3 + out_2, training=training)
        output = self.relu(in_4 + out_2 + out_3)
        return output


class BlockSrc0(tf.keras.layers.Layer):
    """
    [0, 2, 1, 1, 8],  # op_list,
    [[0, 1, 2, 1, 3],
     [1, 2, 3, 4, 4]]  # adj_list
    Conv op at inds: [1] can function as a reduction op
    70.83%/13.59M/2.87234ms
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockSrc0", data_format="channels_last"):
        super(BlockSrc0, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv11_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.conv33_2 = AugmentedConv2D(self.Cout, kernel_size=3, stride=1, padding="same",
                                        activation=tf.keras.layers.ReLU(), name=self._name + "/conv33_2",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.conv33_3 = AugmentedConv2D(self.Cout, kernel_size=3, stride=1, padding="same",
                                        activation=None, name=self._name + "/conv33_3",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        inputs = self.conv11(inputs, training=training)
        out_2 = self.conv33_2(inputs, training=training)
        out_3 = self.conv33_3(out_2, training=training)
        output = self.relu(inputs + out_3)
        return output


class BlockSrc1(tf.keras.layers.Layer):
    """
    [0, 2, 1, 1, 8],
    [[0, 0, 2, 1, 3],
     [1, 2, 3, 4, 4]]
    Conv op at inds: [1, 3] can function as a reduction op
    70.11%/9.72M/2.6289ms
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockSrc1", data_format="channels_last"):
        super(BlockSrc1, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv11_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.conv33_2 = AugmentedConv2D(self.Cin, kernel_size=3, stride=1, padding="same",
                                        activation=tf.keras.layers.ReLU(), name=self._name + "/conv33_2",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.conv33_3 = AugmentedConv2D(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                        activation=None, name=self._name + "/conv33_3",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        shortcut = self.conv11(inputs, training=training)
        out_2 = self.conv33_2(inputs, training=training)
        out_3 = self.conv33_3(out_2, training=training)
        output = self.relu(shortcut + out_3)
        return output


class BlockSrc3(tf.keras.layers.Layer):
    """
    [0, 2, 2, 1, 2, 8],
    [[0, 0, 2, 3, 4, 1],
     [1, 2, 3, 4, 5, 5]]
    Conv op at inds: [1, 3] can function as a reduction op
    68.96%/6.91M/2.3003ms
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockSrc3", data_format="channels_last"):
        super(BlockSrc3, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv11_1 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                        activation=None, name=self._name + "/conv11_1",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.conv11_2 = AugmentedConv2D(self.Cin, kernel_size=1, stride=1, padding="same",
                                        activation=tf.keras.layers.ReLU(), name=self._name + "/conv11_2",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.conv33 = AugmentedConv2D(self.Cout, kernel_size=3, stride=self.stride, padding="same",
                                      activation=tf.keras.layers.ReLU(), name=self._name + "/conv33_3",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.conv11_4 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                        activation=None, name=self._name + "/conv11_4",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        shortcut = self.conv11_1(inputs, training=training)
        out_2 = self.conv11_2(inputs, training=training)
        out_3 = self.conv33(out_2, training=training)
        out_4 = self.conv11_4(out_3, training=training)
        output = self.relu(shortcut + out_4)
        return output


class BlockSrc7(tf.keras.layers.Layer):
    """
    [0, 2, 3, 3, 8],
    [[0, 1, 1, 2, 3],
     [1, 4, 2, 3, 4]]
    Conv op at inds: [1] can function as a reduction op
    70.89%/9.42M/2.9014ms
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockSrc7", data_format="channels_last"):
        super(BlockSrc7, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv11 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                      activation=None, name=self._name + "/conv11_1",
                                      data_format=data_format, use_bias=False, use_bn=True)
        self.inception_s_1 = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_1")
        self.inception_s_2 = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                        data_format=data_format, name=self._name + "/inception_s_2")
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        inputs = self.conv11(inputs, training=training)
        out_2 = self.inception_s_1(inputs, training=training)
        out_3 = self.inception_s_2(out_2, training=training)
        output = self.relu(inputs + out_3)
        return output


class BlockSrc8(tf.keras.layers.Layer):
    """
    [0, 2, 3, 2, 8],
    [[0, 1, 1, 2, 3],
     [1, 4, 2, 3, 4]]
    Conv op at inds: [1] can function as a reduction op
    67.85%/5.94M/2.19756ms
    """
    def __init__(self, C_in, C_out, b_stride=1,
                 name="BlockSrc8", data_format="channels_last"):
        super(BlockSrc8, self).__init__()
        self._name = name
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.conv11_1 = AugmentedConv2D(self.Cout, kernel_size=1, stride=self.stride, padding="same",
                                        activation=None, name=self._name + "/conv11_1",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.inception_s = InceptionS(self.Cout, kernel_size=3, stride=1, padding="same",
                                      data_format=data_format, name=self._name + "/inception_s_2")
        self.conv11_3 = AugmentedConv2D(self.Cout, kernel_size=1, stride=1, padding="same",
                                        activation=None, name=self._name + "/conv11_3",
                                        data_format=data_format, use_bias=False, use_bn=True)
        self.relu = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, inputs, training=True):
        inputs = self.conv11_1(inputs, training=training)
        out_2 = self.inception_s(inputs, training=training)
        out_3 = self.conv11_3(out_2, training=training)
        output = self.relu(inputs + out_3)
        return output
