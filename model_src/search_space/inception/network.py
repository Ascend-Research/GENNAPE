import torch
import torch.nn as nn
from model_src.search_space.inception.constants import C_INIT
from model_src.search_space.inception.block import InceptionBlock
from model_src.search_space.inception.arch_config import NetConfig
from model_src.search_space.inception.operations import ConvBNReLU


class InceptionNet(torch.nn.Module):

    def __init__(self, net_config:NetConfig,
                 C_in=3, C_stem=C_INIT, n_classes=10,
                 bn=True, bias=True,
                 dropout_prob=0.):
        super(InceptionNet, self).__init__()
        self.C_in = C_in
        self.C_stem = C_stem
        self.bn = bn
        self.bias = bias
        self.net_config = net_config

        self.stem = ConvBNReLU(self.C_in, self.C_stem,
                               kernel_size=3, stride=1, padding=1)

        C_curr = self.C_stem
        net_blocks = []
        for si, stage in enumerate(net_config.stages):
            block_config = stage["block"]
            n_blocks = stage["n_blocks"]
            C_out = stage["C_out"]
            stride = stage["stride"]
            for bi in range(n_blocks):
                block = InceptionBlock(block_config, C_curr, C_out,
                                       self.bn, self.bias,
                                       b_stride=stride if bi==0 else 1)
                net_blocks.append(block)
                C_curr = C_out
        self.blocks = nn.Sequential(*net_blocks)

        self.post_conv = ConvBNReLU(C_in=C_curr, C_out=512,
                                    kernel_size=3, stride=1, padding=1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.post_conv(x)
        x = self.global_pooling(x)
        x = self.dropout(x)
        logits = self.classifier(x.view(x.size(0), -1))
        return logits
