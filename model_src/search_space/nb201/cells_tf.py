import tensorflow.compat.v1 as tf
from model_src.search_space.nb201.operations_tf import get_op_model


class Cell(tf.keras.layers.Layer):

    def __init__(self, op_names, op_input_inds, C_in, C_out,
                 data_format="channels_last", name="Cell"):
        super(Cell, self).__init__()
        self._name = name
        self.data_format = data_format
        self.op_names = op_names # Make sure no input/output node label is included
        self.op_input_inds = op_input_inds
        self.C_in = C_in
        self.C_out = C_out
        self.ops = []
        for i, op_name in enumerate(op_names):
            op = get_op_model(op_name, self.C_in, stride=1, data_format=self.data_format,
                              scope_name="{}/{}_{}".format(self._name, op_name, i))
            self.ops.append(op)

    def __len__(self):
        return len(self.ops)

    @property
    def channel_size(self):
        return self.C_in

    def call(self, x, training=True):
        states = [x]
        for opi, op in enumerate(self.ops):
            input_state_inds = self.op_input_inds[opi]
            input_state = sum(states[si] for si in input_state_inds)
            output_state = op.call(input_state, training=training)
            states.append(output_state)
        return states[-1] + states[-2] + states[-3] # Last node in a cell graph


class ReductionCell(tf.keras.layers.Layer):

    def __init__(self, C_in, C_out, stride,
                 data_format="channels_last",
                 name="ReductionCell"):
        super(ReductionCell, self).__init__()
        self._name = name
        self.data_format = data_format
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.conv1 = tf.keras.layers.Conv2D(filters=C_in, kernel_size=(3, 3), strides=(1, 1),
                                            padding="same", data_format=data_format,
                                            activation=None, use_bias=False,
                                            kernel_initializer="he_normal", name=self._name + "/conv1")
        self.act = tf.keras.layers.ReLU(name=self._name + "/relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=C_out, kernel_size=(3, 3), strides=stride,
                                            padding="same", data_format=data_format,
                                            activation=None, use_bias=False,
                                            kernel_initializer="he_normal", name=self._name + "/conv2")
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=stride, padding="same",
                                                        data_format=data_format, name=self._name + "/avgpool")
        self.proj = tf.keras.layers.Conv2D(filters=C_out, kernel_size=(1, 1), strides=(1, 1),
                                           padding="same", data_format=data_format,
                                           activation=None, use_bias=False,
                                           kernel_initializer="he_normal", name=self._name + "/proj")

    @property
    def channel_size(self):
        return self.C_out

    def call(self, x, training=True):
        res_x = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        res_x = self.avgpool(res_x)
        res_x = self.proj(res_x)
        return self.act(x + res_x)
