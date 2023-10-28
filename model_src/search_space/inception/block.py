import torch
import torch.nn as nn
from model_src.search_space.inception.arch_config import BlockConfig
from model_src.search_space.inception.operations import ConvBN, get_op_model
from model_src.search_space.inception.arch_utils import get_boundary_conv_op_idx, get_boundary_C_change_op_idx


class InceptionBlock(nn.Module):

    def __init__(self, config:BlockConfig, C_in, C_out,
                 bn=True, bias=True, b_stride=1):
        super(InceptionBlock, self).__init__()
        self.config = config
        self.bn = bn
        self.bias = bias
        self.C_in = C_in
        self.C_out = C_out
        self.b_stride = b_stride
        self.paths = nn.ModuleList()
        self.C_proj = None
        self._build_from_config(self.config)

    @staticmethod
    def _build_path(op_names, op_types,
                    C_in, C_internal, C_out,
                    b_stride, bn, bias):
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
                                  bn=bn, bias=bias)
                C_curr = C_internal
            elif op_i == first_conv_idx == last_C_change_idx:
                op = get_op_model(C_curr, C_out, stride=b_stride, op_name=op_name,
                                  bn=bn, bias=bias)
                C_curr = C_out
            elif op_i == first_C_change_idx:
                op = get_op_model(C_curr, C_internal, stride=1, op_name=op_name,
                                  bn=bn, bias=bias)
                C_curr = C_internal
            elif op_i == first_conv_idx:
                op = get_op_model(C_curr, C_curr, stride=b_stride, op_name=op_name,
                                  bn=bn, bias=bias)
            elif op_i == last_C_change_idx:
                op = get_op_model(C_curr, C_out, stride=1, op_name=op_name,
                                  bn=bn, bias=bias)
                C_curr = C_out
            else:
                op = get_op_model(C_curr, C_curr, stride=1, op_name=op_name,
                                  bn=bn, bias=bias)
            path.append(op)
        assert C_curr == C_out
        return nn.Sequential(*path)

    def _build_from_config(self, config:BlockConfig):
        C_out_vals = []
        for p in config.paths:
            ops = p["ops"]
            op_types = p["op_types"]
            C_internal = p["C_internal"]
            C_out = p["C_out"]
            path = self._build_path(ops, op_types, self.C_in, C_internal, C_out,
                                    self.b_stride, self.bn, self.bias)
            C_out_vals.append(C_out)
            self.paths.append(path)

        cat_C_out = sum(C_out_vals)
        if cat_C_out != self.C_out:
            self.C_proj = ConvBN(cat_C_out, self.C_out,
                                 kernel_size=1, stride=1, padding=0,
                                 bn=self.bn, bias=self.bias)

    def forward(self, x):
        res_x = x
        p_outs = []
        for path in self.paths:
            p_out = path(x)
            p_outs.append(p_out)
        if len(p_outs) > 1:
            out = torch.cat(p_outs, dim=1)
        else:
            out = p_outs[0]
        if self.C_proj is not None:
            out = self.C_proj(out)
        if self.b_stride == 1 and self.C_in == self.C_out:
            out = out + res_x
        assert out.shape[1] == self.C_out
        return out
