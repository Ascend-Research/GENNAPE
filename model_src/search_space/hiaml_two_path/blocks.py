import torch.nn as nn
from model_src.search_space.hiaml_two_path.operations import get_two_path_op_model


class SinglePathBlock(nn.Module):

    def __init__(self, op_names, op_types, C_in, C_out,
                 get_op_model=get_two_path_op_model,
                 b_stride=1):
        super(SinglePathBlock, self).__init__()
        self.op_names = op_names
        self.op_types = op_types
        self.C_in = C_in
        self.C_out = C_out
        self.b_stride = b_stride
        self.ops = nn.ModuleList()
        C_curr = C_in
        is_first_conv = True
        for op_name, op_type in zip(op_names, op_types):
            if op_type == "conv" and is_first_conv:
                stride = b_stride
                op = get_op_model(C_curr, C_out, stride=stride, op_name=op_name)
                C_curr = C_out
                is_first_conv = False
            else:
                op = get_op_model(C_curr, C_curr, stride=1, op_name=op_name)
            self.ops.append(op)

    def __len__(self):
        return len(self.ops)

    @property
    def channel_size(self):
        return self.C_out

    def forward(self, x):
        res_x = x
        for op in self.ops:
            x = op(x)
        if self.b_stride == 1 and self.C_in == self.C_out:
            x = x + res_x
        return x
