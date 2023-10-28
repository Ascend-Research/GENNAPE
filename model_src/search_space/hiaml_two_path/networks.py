import torch
import torch.nn as nn
from model_src.search_space.hiaml_two_path.arch_configs import TwoPathNetConfig
from model_src.search_space.hiaml_two_path.blocks import SinglePathBlock
from model_src.search_space.hiaml_two_path.constants import HIAML_NUM_BLOCKS_PER_STACK
from model_src.search_space.hiaml_two_path.operations import ConvBNReLU, get_hiaml_block_op_model


class TwoPathNet(torch.nn.Module):

    def __init__(self, net_config:TwoPathNetConfig,
                 C_in=3, C_stem=32, n_classes=10,
                 dropout_prob=0.):
        super(TwoPathNet, self).__init__()
        self.C_in = C_in
        self.C_stem = C_stem
        self.net_config = net_config

        # Stem part
        self.stem_proj = ConvBNReLU(self.C_in, self.C_stem,
                                    kernel_size=3, stride=1, padding=1)
        self.stem_reduce = ConvBNReLU(self.C_stem, self.C_stem,
                                      kernel_size=3, stride=2, padding=1)

        # Build first path of blocks
        C_in = self.C_stem
        self.path1 = nn.ModuleList()
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
                                    C_in, C_out, b_stride=stride)
            C_in = C_out
            self.path1.append(block)

        # Build second stage of blocks
        C_in = self.C_stem
        self.path2 = nn.ModuleList()
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
                                    C_in, C_out, b_stride=stride)
            C_in = C_out
            self.path2.append(block)

        # Output part
        self.post_conv = ConvBNReLU(C_in=256, C_out=512,
                                    kernel_size=3, stride=1, padding=1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):

        x = self.stem_proj(x)
        x = self.stem_reduce(x)

        x1 = x
        for b in self.path1:
            x1 = b(x1)

        x2 = x
        for b in self.path2:
            x2 = b(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.post_conv(x)
        x = self.global_pooling(x)
        x = self.dropout(x)
        logits = self.classifier(x.view(x.size(0), -1))

        return logits


class HiAMLStem(torch.nn.Module):

    def __init__(self, C_in=3, C_out=64,
                 kernel_size=3, stride=2):
        super(HiAMLStem, self).__init__()
        self.conv = torch.nn.Conv2d(C_in,
                                    out_channels=C_out, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size//2, bias=False)
        self.bn = torch.nn.BatchNorm2d(C_out, affine=True)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class HiAMLOutNet(torch.nn.Module):
    """Re-implemented from ResnetPostNet"""
    def __init__(self, C_in, n_classes):
        super(HiAMLOutNet, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_in, n_classes)

    def forward(self, x):
        out = self.global_pooling(x)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class HiAMLNet(torch.nn.Module):

    def __init__(self, net_config, # Same block in every stack
                 C_in=3, C_init=64, stem_kernel_size=3, stem_stride=2,
                 reduction_stage_inds=(1, 3), n_classes=10,
                 n_blocks_per_stack=HIAML_NUM_BLOCKS_PER_STACK):
        super(HiAMLNet, self).__init__()
        self.reduction_stage_inds = reduction_stage_inds
        self.stem = HiAMLStem(C_in=C_in, C_out=C_init,
                              kernel_size=stem_kernel_size,
                              stride=stem_stride)
        self.blocks = nn.ModuleList()
        C_block = C_init
        for si, op_name in enumerate(net_config):
            for bi in range(n_blocks_per_stack):
                if si in reduction_stage_inds and bi == 0:
                    # Reduction block
                    new_C_block = 2 * C_block
                    block = get_hiaml_block_op_model(C_block, new_C_block,
                                                     stride=2, op_name=op_name)
                    C_block = new_C_block
                else:
                    # Regular block
                    block = get_hiaml_block_op_model(C_block, C_block,
                                                     stride=1, op_name=op_name)
                self.blocks.append(block)
        self.out_net = HiAMLOutNet(C_block, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for bi, block in enumerate(self.blocks):
            x = block(x)
        return self.out_net(x)
