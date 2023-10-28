import tensorflow.compat.v1 as tf
from model_src.search_space.hiaml_two_path.arch_configs import TwoPathNetConfig
from model_src.search_space.hiaml_two_path.operations_tf import AugmentedConv2D
from model_src.search_space.hiaml_two_path.constants import HIAML_NUM_BLOCKS_PER_STACK
from model_src.search_space.hiaml_two_path.blocks_tf import SinglePathBlock
from model_src.search_space.hiaml_two_path.operations_tf import BN_MOMENTUM, BN_EPSILON, get_hiaml_block_op_model


class TwoPathNet(tf.keras.Model):
    """
    This implementation is supposed to match the torch version as much as possible
    """
    def __init__(self, net_config:TwoPathNetConfig,
                 C_in=3, C_stem=32, n_classes=10,
                 dropout_prob=0., name="TwoPathNet",
                 data_format="channels_last"):
        super(TwoPathNet, self).__init__()
        assert data_format == "channels_last"
        self._name = name
        self.C_in = C_in
        self.C_stem = C_stem
        self.net_config = net_config
        self.data_format = data_format

        self.stem_proj = AugmentedConv2D(C_stem,
                                         kernel_size=3, stride=1, padding="same",
                                         activation=tf.keras.layers.ReLU(max_value=6),
                                         data_format=self.data_format,
                                         name="{}/StemProj".format(self._name))
        self.stem_reduce = AugmentedConv2D(C_stem,
                                           kernel_size=3, stride=2, padding="same",
                                           activation=tf.keras.layers.ReLU(max_value=6),
                                           data_format=self.data_format,
                                           name="{}/StemReduce".format(self._name))

        # Build first path of blocks
        C_in = self.C_stem
        self.path1 = []
        for bi in range(len(self.net_config.path1)):
            if bi in self.net_config.path1_reduce_inds:
                stride = 2
                C_out = C_in * 2
            else:
                stride = 1
                C_out = C_in
            block_config = self.net_config.path1[bi]
            block = SinglePathBlock(block_config.op_names,
                                    block_config.op_types,
                                    C_in, C_out, b_stride=stride,
                                    name="{}/SinglePathBlock_p1_b{}".format(self._name, bi))
            C_in = C_out
            self.path1.append(block)

        # Build second stage of blocks
        C_in = self.C_stem
        self.path2 = []
        for bi in range(len(self.net_config.path2)):
            if bi in self.net_config.path2_reduce_inds:
                stride = 2
                C_out = C_in * 2
            else:
                stride = 1
                C_out = C_in
            block_config = self.net_config.path2[bi]
            block = SinglePathBlock(block_config.op_names,
                                    block_config.op_types,
                                    C_in, C_out, b_stride=stride,
                                    name = "{}/SinglePathBlock_p2_b{}".format(self._name, bi))
            C_in = C_out
            self.path2.append(block)

        # Output part
        self.post_conv = AugmentedConv2D(512, kernel_size=3, stride=1, padding="same",
                                         activation=tf.keras.layers.ReLU(max_value=6),
                                         data_format=self.data_format,
                                         name="{}/PostConv".format(self._name))
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format, name=self._name + "/GP")
        self.dropout = tf.keras.layers.Dropout(dropout_prob, name=self._name + "/Dropout")
        self.classifier = tf.keras.layers.Dense(n_classes, name=self._name + "/FC")

    def call(self, inputs, training=True):
        x = self.stem_proj(inputs, training=training)
        x = self.stem_reduce(x, training=training)

        x1 = x
        for b in self.path1:
            x1 = b(x1, training=training)

        x2 = x
        for b in self.path2:
            x2 = b(x2, training=training)

        x = tf.concat([x1, x2], axis=-1)
        x = self.post_conv(x, training=training)
        x = self.global_pooling(x)
        x = self.dropout(x, training=training)
        logits = self.classifier(x)
        return logits


class HiAMLStem(tf.keras.layers.Layer):

    def __init__(self, C_out=64, kernel_size=3, stride=2,
                 name="HiAMLStem", data_format="channels_last"):
        super(HiAMLStem, self).__init__()
        self._name = name
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv = tf.keras.layers.Conv2D(filters=C_out, kernel_size=kernel_size, strides=stride,
                                           padding="same", data_format=data_format,
                                           activation=None, use_bias=False,
                                           kernel_initializer="he_normal", name=self._name + "/conv")
        self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis,
                                                     momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                     name=self._name + "/bn")
        self.act = tf.keras.layers.ReLU(name=self._name + "/relu")

    def call(self, x, training=True):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x


class HiAMLOutNet(tf.keras.layers.Layer):

    def __init__(self, n_classes,
                 name="HiAMLOutNet",
                 data_format="channels_last"):
        super(HiAMLOutNet, self).__init__()
        self._name = name
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format, name=self._name + "/GP")
        self.classifier = tf.keras.layers.Dense(n_classes, name=self._name + "/FC")

    def call(self, x, training=True):
        out = self.global_pooling(x)
        logits = self.classifier(out)
        return logits


class HiAMLNet(tf.keras.Model):
    """
    This implementation is supposed to match the torch version as much as possible
    """
    def __init__(self, net_config, # Same block in every stack
                 C_init=64, stem_kernel_size=3, stem_stride=2,
                 reduction_stage_inds=(1, 3), n_classes=10,
                 n_blocks_per_stack=HIAML_NUM_BLOCKS_PER_STACK,
                 name="HiAMLNet", data_format="channels_last"):
        super(HiAMLNet, self).__init__()
        self._name = name
        self.data_format = data_format
        self.reduction_stage_inds = reduction_stage_inds
        self.stem = HiAMLStem(C_out=C_init,
                              kernel_size=stem_kernel_size,
                              stride=stem_stride,
                              name="{}/Stem".format(self._name),
                              data_format=self.data_format)
        self.blocks = []
        C_block = C_init
        for si, op_name in enumerate(net_config):
            for bi in range(n_blocks_per_stack):
                if si in reduction_stage_inds and bi == 0:
                    # Reduction block
                    new_C_block = 2 * C_block
                    block = get_hiaml_block_op_model(C_block, new_C_block,
                                                     stride=2, op_name=op_name,
                                                     scope_name="{}/HiAMLBlock_{}_{}_{}".format(self._name, si,
                                                                                                bi, op_name),
                                                     data_format=self.data_format)
                    C_block = new_C_block
                else:
                    # Regular block
                    block = get_hiaml_block_op_model(C_block, C_block,
                                                     stride=1, op_name=op_name,
                                                     scope_name="{}/HiAMLBlock_{}_{}_{}".format(self._name, si,
                                                                                                bi, op_name),
                                                     data_format=self.data_format)
                self.blocks.append(block)
        self.out_net = HiAMLOutNet(n_classes, name="{}/OutNet".format(self._name))

    def call(self, x, training=True):
        x = self.stem(x, training=training)
        for bi, block in enumerate(self.blocks):
            x = block(x, training=training)
        return self.out_net(x, training=training)
