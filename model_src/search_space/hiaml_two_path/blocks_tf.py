import tensorflow.compat.v1 as tf
from model_src.search_space.hiaml_two_path.operations_tf import get_two_path_op_model


class SinglePathBlock(tf.keras.layers.Layer):

    def __init__(self, op_names, op_types, C_in, C_out,
                 get_op_model=get_two_path_op_model,
                 b_stride=1, name="SinglePathBlock"):
        super(SinglePathBlock, self).__init__()
        self._name = name
        self.op_names = op_names
        self.op_types = op_types
        self.C_in = C_in
        self.C_out = C_out
        self.b_stride = b_stride
        self.ops = []
        C_curr = C_in
        is_first_conv = True
        for i, (op_name, op_type) in enumerate(zip(op_names, op_types)):
            if op_type == "conv" and is_first_conv:
                stride = b_stride
                op = get_op_model(C_curr, C_out, stride=stride, op_name=op_name,
                                  scope_name="{}/{}_{}".format(self._name, op_name, i))
                C_curr = C_out
                is_first_conv = False
            else:
                op = get_op_model(C_curr, C_curr, stride=1, op_name=op_name,
                                  scope_name="{}/{}_{}".format(self._name, op_name, i))
            self.ops.append(op)

    def __len__(self):
        return len(self.ops)

    @property
    def channel_size(self):
        return self.C_in

    def call(self, x, training=True):
        x_in = x
        for op in self.ops:
            x = op.call(x, training=training)
        if all(v1 == v2 for v1, v2 in zip(x.shape, x_in.shape)):
            x = x + x_in
        return x
