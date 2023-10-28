import tensorflow.compat.v1 as tf
from model_src.search_space.inception.constants import C_INIT
from model_src.search_space.inception.arch_config import NetConfig
from model_src.search_space.inception.block_tf import InceptionBlock
from model_src.search_space.inception.operations_tf import AugmentedConv2D


class InceptionNet(tf.keras.Model):

    def __init__(self, net_config:NetConfig,
                 C_in=3, C_stem=C_INIT, n_classes=10,
                 use_bn=True, use_bias=True,
                 name="InceptionNet",
                 data_format="channels_last",
                 dropout_prob=0.):
        super(InceptionNet, self).__init__()
        self._name = name
        self.C_in = C_in
        self.C_stem = C_stem
        self.use_bn = use_bn
        self.use_bias = use_bias
        self.net_config = net_config
        self.data_format = data_format

        self.stem = AugmentedConv2D(filters=self.C_stem, kernel_size=3, stride=1, padding="same",
                                    activation=tf.keras.layers.ReLU(max_value=6, name=self._name + "/stem/relu6"),
                                    name=self._name + "/stem", use_bn=self.use_bn, use_bias=self.use_bias,
                                    data_format=self.data_format)

        C_curr = self.C_stem
        self.net_blocks = []
        for si, stage in enumerate(net_config.stages):
            block_config = stage["block"]
            n_blocks = stage["n_blocks"]
            C_out = stage["C_out"]
            stride = stage["stride"]
            for bi in range(n_blocks):
                block = InceptionBlock(block_config, C_curr, C_out,
                                       use_bn=self.use_bn, use_bias=self.use_bias,
                                       b_stride=stride if bi==0 else 1,
                                       name="{}/Block_s{}_b{}".format(self._name, si, bi),
                                       data_format=self.data_format)
                self.net_blocks.append(block)
                C_curr = C_out

        self.post_conv = AugmentedConv2D(filters=512, kernel_size=3, stride=1, padding="same",
                                         activation=tf.keras.layers.ReLU(max_value=6,
                                                                         name=self._name + "/post_conv/relu6"),
                                         name=self._name + "/post_conv", use_bn=self.use_bn, use_bias=self.use_bias,
                                         data_format=self.data_format)
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format, name=self._name + "/GP")
        self.dropout = tf.keras.layers.Dropout(dropout_prob, name=self._name + "/Dropout")
        self.classifier = tf.keras.layers.Dense(n_classes, name=self._name + "/FC")

    def call(self, x, training=True):
        x = self.stem(x, training=training)
        for block in self.net_blocks:
            x = block.call(x, training=training)
        x = self.post_conv(x, training=training)
        x = self.global_pooling(x)
        x = self.dropout(x, training=training)
        logits = self.classifier(x)
        return logits
