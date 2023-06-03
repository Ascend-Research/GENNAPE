import tensorflow.compat.v1 as tf
from model_src.search_space.nb201.cells_tf import Cell, ReductionCell
from model_src.search_space.nb201.operations_tf import BN_MOMENTUM, BN_EPSILON


class Stem(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=3, stride=1, padding="same", name="NB201Stem",
                 data_format="channels_last", use_bias=False):
        super(Stem, self).__init__()
        self._name = name
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                           padding=padding, data_format=data_format, activation=None,
                                           use_bias=use_bias, kernel_initializer="he_normal",
                                           name=self._name + "/conv")
        self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis,
                                                     momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                     name=self._name + "/bn")

    def call(self, x, training=True):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


class OutNet(tf.keras.layers.Layer):

    def __init__(self, n_classes,
                 name="NB201OutNet",
                 data_format="channels_last"):
        super(OutNet, self).__init__()
        self._name = name
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format, name=self._name + "/GP")
        self.dropout = tf.keras.layers.Dropout(0.0, name=self._name + "/Dropout")
        self.classifier = tf.keras.layers.Dense(n_classes, name=self._name + "/FC")

    def call(self, x, training=True):
        out = self.global_pooling(x)
        out = self.dropout(out, training=training)
        logits = self.classifier(out)
        return logits


class NB201Net(tf.keras.Model):

    def __init__(self, op_names, op_input_inds,
                 n=5, n_stacks=3, C_init=16, n_classes=10,
                 name="NB201Net", data_format="channels_last"):
        super(NB201Net, self).__init__()
        self._name = name
        self.data_format = data_format
        self.stem = Stem(C_init,
                         data_format=self.data_format,
                         name="{}/Stem".format(self._name))
        self.cells = []
        C_curr = C_init

        for si in range(n_stacks):

            for ci in range(n):
                cell = Cell(op_names, op_input_inds, C_curr, C_curr,
                            data_format=self.data_format,
                            name="{}/Cell_{}_{}".format(self._name, si, ci))
                self.cells.append(cell)

            if si != n_stacks - 1:
                reduction_cell = ReductionCell(C_curr, 2 * C_curr, 2,
                                               data_format=self.data_format,
                                               name="{}/ReductionCell_{}".format(self._name, si))
                self.cells.append(reduction_cell)
                C_curr *= 2

        self.out_net = OutNet(n_classes,
                              data_format=self.data_format,
                              name="{}/OutNet".format(self._name))

    def call(self, inputs, training=True):
        x = self.stem(inputs, training=training)
        for bi, cell in enumerate(self.cells):
            x = cell.call(x, training=training)
        x = self.out_net(x, training=training)
        return x
