import tensorflow as tf
from model_src.search_space.inception.arch_config import BlockConfig
from model_src.search_space.inception.operations_tf import AugmentedConv2D, get_op_model
from model_src.search_space.inception.arch_utils import get_boundary_conv_op_idx, get_boundary_C_change_op_idx


class InceptionBlock(tf.keras.layers.Layer):

    def __init__(self, config:BlockConfig, C_in, C_out,
                 use_bn=True, use_bias=True,
                 b_stride=1, name="InceptionBlock",
                 data_format="channels_last"):
        super(InceptionBlock, self).__init__()
        self._name = name
        self.data_format = data_format
        self.config = config
        self.C_in = C_in
        self.C_out = C_out
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.b_stride = b_stride
        self.paths = []
        self.C_proj = None
        self._build_from_config(self.config)

    @staticmethod
    def _build_path(block_name, op_names, op_types,
                    C_in, C_internal, C_out,
                    b_stride, use_bn, use_bias,
                    data_format="channels_last"):
        C_curr = C_in
        path = []
        first_C_change_idx = get_boundary_C_change_op_idx(op_names, op_types)
        first_conv_idx = get_boundary_conv_op_idx(op_types)
        last_C_change_idx = get_boundary_C_change_op_idx(op_names, op_types, reverse=True)
        last_conv_idx = get_boundary_conv_op_idx(op_types, reverse=True)
        assert first_C_change_idx <= first_conv_idx
        assert last_conv_idx <= last_C_change_idx
        if C_internal != C_out:
            assert first_C_change_idx != last_C_change_idx, "Invalid config: {}".format(op_names)

        # First C change capable op handles C reduction
        # First conv op handles HW reduction
        # Last C change capable op handles C resetting
        for op_i, op_name in enumerate(op_names):
            if op_i == first_C_change_idx == first_conv_idx:
                op = get_op_model(C_curr, C_internal, stride=b_stride, op_name=op_name,
                                  scope_name="{}/{}_{}".format(block_name, op_name, op_i),
                                  use_bn=use_bn, use_bias=use_bias,
                                  data_format=data_format)
                C_curr = C_internal
            elif op_i == first_conv_idx == last_C_change_idx:
                op = get_op_model(C_curr, C_out, stride=b_stride, op_name=op_name,
                                  scope_name="{}/{}_{}".format(block_name, op_name, op_i),
                                  use_bn=use_bn, use_bias=use_bias,
                                  data_format=data_format)
                C_curr = C_out
            elif op_i == first_C_change_idx:
                op = get_op_model(C_curr, C_internal, stride=1, op_name=op_name,
                                  scope_name="{}/{}_{}".format(block_name, op_name, op_i),
                                  use_bn=use_bn, use_bias=use_bias,
                                  data_format=data_format)
                C_curr = C_internal
            elif op_i == first_conv_idx:
                op = get_op_model(C_curr, C_curr, stride=b_stride, op_name=op_name,
                                  scope_name="{}/{}_{}".format(block_name, op_name, op_i),
                                  use_bn=use_bn, use_bias=use_bias,
                                  data_format=data_format)
            elif op_i == last_C_change_idx:
                op = get_op_model(C_curr, C_out, stride=1, op_name=op_name,
                                  scope_name="{}/{}_{}".format(block_name, op_name, op_i),
                                  use_bn=use_bn, use_bias=use_bias,
                                  data_format=data_format)
                C_curr = C_out
            else:
                op = get_op_model(C_curr, C_curr, stride=1, op_name=op_name,
                                  scope_name="{}/{}_{}".format(block_name, op_name, op_i),
                                  use_bn=use_bn, use_bias=use_bias,
                                  data_format=data_format)
            path.append(op)
        assert C_curr == C_out
        return path

    def _build_from_config(self, config:BlockConfig):
        C_out_vals = []
        for pi, p in enumerate(config.paths):
            ops = p["ops"]
            op_types = p["op_types"]
            C_internal = p["C_internal"]
            C_out = p["C_out"]
            path = self._build_path(self._name + "/path_{}".format(pi),
                                    ops, op_types, self.C_in, C_internal, C_out,
                                    self.b_stride, self.use_bn, self.use_bias,
                                    self.data_format)
            C_out_vals.append(C_out)
            self.paths.append(path)

        cat_C_out = sum(C_out_vals)
        if cat_C_out != self.C_out:
            self.C_proj = AugmentedConv2D(self.C_out,
                                          kernel_size=1, stride=1, padding="same",
                                          activation=None, name="{}/C_proj".format(self._name),
                                          use_bn=self.use_bn, use_bias=self.use_bias,
                                          data_format=self.data_format)

    def call(self, x, training=True):
        res_x = x
        p_outs = []
        for path in self.paths:
            curr = x
            for op in path:
                curr = op.call(curr, training=training)
            p_outs.append(curr)
        if len(p_outs) > 1:
            out = tf.concat(p_outs, -1)
        else:
            out = p_outs[0]
        if self.C_proj is not None:
            out = self.C_proj(out, training=training)
        if self.b_stride == 1 and self.C_in == self.C_out:
            out = out + res_x
        return out
