import torch
import torch.nn as nn


def _decode_kernel_padding_size(op):
    k = [int(s) for s in op.split("x")]
    assert all(v % 2 != 0 for v in k), "Only odd kernel sizes supported"
    padding = [v // 2 for v in k]
    return k, padding


def _decode_int_value(op, code="e"):
    assert op.startswith(code)
    op = op.replace(code, "")
    ratio = int(op)
    return ratio


def get_two_path_op_model(C_in, C_out, stride, op_name):
    if op_name.startswith("conv"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k, padding = _decode_kernel_padding_size(conv_op_str.replace("conv", ""))
        if len(parts) == 1:
            op = ConvBNReLU(C_in, C_out, kernel_size=k, stride=stride,
                            padding=padding)
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = ConvBNReLU(C_hidden, C_hidden, kernel_size=k, stride=stride,
                                      padding=padding)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op)
            elif opt.startswith("e"):
                e = _decode_int_value(opt, "e")
                C_hidden = int(C_in * e)
                inner_op = ConvBNReLU(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0])
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op)
            else:
                raise ValueError
        else:
            raise ValueError

    elif op_name.startswith("inception_s"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k, padding = _decode_kernel_padding_size(conv_op_str.replace("inception_s", ""))
        if len(parts) == 1:
            op = InceptionS(C_in, C_out, kernel_size=k[0], stride=stride,
                            padding=padding[0])
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = InceptionS(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0])
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op)
            elif opt.startswith("e"):
                e = _decode_int_value(opt, "e")
                C_hidden = int(C_in * e)
                inner_op = InceptionS(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0])
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op)
            else:
                raise ValueError
        else:
            raise ValueError

    elif op_name.startswith("double"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k, padding = _decode_kernel_padding_size(conv_op_str.replace("double", ""))
        if len(parts) == 1:
            op = nn.Sequential(
                ConvBNReLU(C_in, C_in, kernel_size=k, stride=stride,
                           padding=padding),
                ConvBNReLU(C_in, C_out, kernel_size=k, stride=1,
                           padding=padding),
            )
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = nn.Sequential(
                    ConvBNReLU(C_hidden, C_hidden, kernel_size=k, stride=stride,
                               padding=padding),
                    ConvBNReLU(C_hidden, C_hidden, kernel_size=k, stride=1,
                               padding=padding),
                )
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op)
            else:
                raise ValueError
        else:
            raise ValueError

    elif op_name.startswith("bottleneck"):
        parts = op_name.split("-")
        if len(parts) == 3:
            _, opt1, opt2 = parts
            if opt1.startswith("r"):
                r = _decode_int_value(opt1, "r")
                C_hidden = max(1, C_in // r)
            elif opt1.startswith("e"):
                e = _decode_int_value(opt1, "e")
                C_hidden = int(C_in * e)
            else:
                raise ValueError
            k = _decode_int_value(opt2, "k")
            op = Bottleneck(C_in, C_hidden, C_out, kernel_size=k, stride=stride, padding=k//2)
        else:
            raise ValueError

    elif op_name.startswith("avgpool"):
        k, padding = _decode_kernel_padding_size(op_name.replace("avgpool", ""))
        op = AvgPool(C_in, C_out, kernel_size=k, stride=stride, padding=padding)

    elif op_name.startswith("maxpool"):
        k, padding = _decode_kernel_padding_size(op_name.replace("maxpool", ""))
        op = MaxPool(C_in, C_out, kernel_size=k, stride=stride, padding=padding)

    else:
        raise ValueError("Unknown op name: {}".format(op_name))
    return op


def get_hiaml_block_op_model(C_in, C_out, stride, op_name):
    if op_name == "hiaml_a22":
        op = BlockA22(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_z":
        op = BlockZ(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_j30":
        op = BlockJ30(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_b23":
        op = BlockB23(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_g30":
        op = BlockG30(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_r30":
        op = BlockR30(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_l20":
        op = BlockL20(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_t20":
        op = BlockT20(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_t31":
        op = BlockT31(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_src0":
        op = BlockSrc0(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_src1":
        op = BlockSrc1(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_src3":
        op = BlockSrc3(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_src7":
        op = BlockSrc7(C_in, C_out, b_stride=stride)

    elif op_name == "hiaml_src8":
        op = BlockSrc8(C_in, C_out, b_stride=stride)

    else:
        raise ValueError("Unknown op name: {}".format(op_name))
    return op


class ConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.op(x)


class ConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ChannelResizeWrapper(nn.Module):

    def __init__(self, C_in, C_hidden, C_out, op):
        super(ChannelResizeWrapper, self).__init__()
        self.proj1 = ConvBN(C_in, C_hidden, 1, 1, 0)
        self.op = op
        self.proj2 = ConvBN(C_hidden, C_out, 1, 1, 0)

    def forward(self, x):
        return self.proj2(self.op(self.proj1(x)))


class Bottleneck(nn.Module):

    def __init__(self, C_in, C_hidden, C_out, kernel_size, stride, padding, affine=True):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBNReLU(C_in, C_hidden, kernel_size, stride, padding, affine=affine)
        self.conv2 = ConvBNReLU(C_hidden, C_out, kernel_size, 1, padding, affine=affine)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class InceptionS(nn.Module):

    def __init__(self, Cin, Cout, kernel_size=3, stride=1, padding=1, affine=True):
        super(InceptionS, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(Cin, Cout, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=False),
            nn.BatchNorm2d(Cout, affine=affine),
            nn.ReLU6(),
            nn.Conv2d(Cout, Cout, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), bias=False),
            nn.BatchNorm2d(Cout, affine=affine),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.op(x)


class AvgPool(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(AvgPool, self).__init__()
        assert C_in == C_out
        self.op = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride, padding),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class MaxPool(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(MaxPool, self).__init__()
        assert C_in == C_out
        self.op = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride, padding),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DepthwiseConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        assert C_in == C_out, "DepthwiseConv does not support channel size change for now"
        super(DepthwiseConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.BatchNorm2d(C_in, affine=affine, eps=1e-3),
            nn.ReLU6(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class SqueezeExcite(nn.Module):

    def __init__(self, C_in, C_squeeze):
        super(SqueezeExcite, self).__init__()
        self.squeeze_conv = nn.Conv2d(C_in, C_squeeze, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU6(inplace=False)
        self.excite_conv = nn.Conv2d(C_squeeze, C_in, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x_sq = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x_sq = self.squeeze_conv(x_sq)
        x_sq = self.relu(x_sq)
        x_sq = self.excite_conv(x_sq)
        return torch.sigmoid(x_sq) * x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, *args):
        return x


class BlockA22(nn.Module):
    """
    BlockState[5]:input,conv3x3,conv1x1,conv1x1,output|Edges: [(0, 1), (1, 2), (1, 4), (2, 3), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockA22, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv33 = ConvBNReLU(self.Cin, self.Cout, kernel_size=3, stride=self.stride, padding=1, affine=self.affine)
        self.conv11_2 = ConvBNReLU(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.conv11_3 = ConvBN(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        out_1 = self.conv33(inputs)
        out_2 = self.conv11_2(out_1)
        out_3 = self.conv11_3(out_2)
        output = self.relu(out_1 + out_3)
        return output


class BlockZ(nn.Module):
    """
    BlockState[4]:input,conv3x3,inception_s,output|Edges: [(0, 1), (1, 2), (1, 3), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockZ, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv33 = ConvBNReLU(self.Cin, self.Cout, kernel_size=3, stride=self.stride, padding=1, affine=self.affine)
        self.inception_s = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        out_1 = self.conv33(inputs)
        out_2 = self.inception_s(out_1)
        output = self.relu(out_1 + out_2)
        return output


class BlockB23(nn.Module):
    """
    BlockState[4]:input,inception_s,inception_s,output|Edges: [(0, 1), (0, 3), (1, 2), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockB23, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.inception_s_1 = InceptionS(self.Cin, self.Cout, stride=self.stride, affine=self.affine)
        self.inception_s_2 = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_3 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0,
                                       affine=self.affine)
        else:
            self.input_proj_3 = None
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        if self.input_proj_3 is not None:
            in_3 = self.input_proj_3(inputs)
        else:
            in_3 = inputs
        out_1 = self.inception_s_1(inputs)
        out_2 = self.inception_s_2(out_1)
        output = self.relu(in_3 + out_2)
        return output


class BlockJ30(nn.Module):
    """
    BlockState[4]:input,conv3x3,inception_s,output|Edges: [(0, 1), (0, 3), (1, 2), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockJ30, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv33 = ConvBNReLU(self.Cin, self.Cout, kernel_size=3, stride=self.stride, padding=1, affine=self.affine)
        self.inception_s = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_3 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0,
                                       affine=self.affine)
        else:
            self.input_proj_3 = None
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        if self.input_proj_3 is not None:
            in_3 = self.input_proj_3(inputs)
        else:
            in_3 = inputs
        out_1 = self.conv33(inputs)
        out_2 = self.inception_s(out_1)
        output = self.relu(in_3 + out_2)
        return output


class BlockG30(nn.Module):
    """
    BlockState[5]:input,inception_s,inception_s,conv1x1,output|
    Edges: [(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockG30, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.inception_s_1 = InceptionS(self.Cin, self.Cout, stride=self.stride, affine=self.affine)
        self.inception_s_2 = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        self.conv11 = ConvBN(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_2 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0,
                                       affine=self.affine)
        else:
            self.input_proj_2 = None
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        if self.input_proj_2 is not None:
            in_2 = self.input_proj_2(inputs)
        else:
            in_2 = inputs
        out_1 = self.inception_s_1(inputs)
        out_2 = self.inception_s_2(in_2 + out_1)
        out_3 = self.conv11(out_2)
        output = self.relu(out_1 + out_3)
        return output


class BlockR30(nn.Module):
    """
    BlockState[5]:input,inception_s,conv1x1,inception_s,output|
    Edges: [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockR30, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.inception_s_1 = InceptionS(self.Cin, self.Cout, stride=self.stride, affine=self.affine)
        self.conv11 = ConvBN(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.inception_s_3 = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_2 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0,
                                       affine=self.affine)
            self.input_proj_4 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0,
                                       affine=self.affine)
        else:
            self.input_proj_2 = None
            self.input_proj_4 = None
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        if self.input_proj_2 is not None:
            in_2 = self.input_proj_2(inputs)
            in_4 = self.input_proj_4(inputs)
        else:
            in_2 = inputs
            in_4 = inputs
        out_1 = self.inception_s_1(inputs)
        out_2 = self.conv11(in_2 + out_1)
        out_3 = self.inception_s_3(out_1)
        output = self.relu(in_4 + out_2 + out_3)
        return output


class BlockL20(nn.Module):
    """
    BlockState[4]:input,conv3x3,conv1x1,output|Edges: [(0, 1), (1, 2), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockL20, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv33 = ConvBNReLU(self.Cin, self.Cout, kernel_size=3, stride=self.stride, padding=1, affine=self.affine)
        self.conv11 = ConvBN(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        out_1 = self.conv33(inputs)
        out_2 = self.conv11(out_1)
        output = self.relu(out_2)
        return output


class BlockT20(nn.Module):
    """
    BlockState[4]:input,conv1x1,inception_s,output|Edges: [(0, 1), (0, 2), (1, 3), (2, 3)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockT20, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv11 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0, affine=self.affine)
        self.inception_s = InceptionS(self.Cin, self.Cout, stride=self.stride, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        out_1 = self.conv11(inputs)
        out_2 = self.inception_s(inputs)
        output = self.relu(out_1 + out_2)
        return output


class BlockT31(nn.Module):
    """
    BlockState[5]:input,inception_s,conv1x1,inception_s,output|
    Edges: [(0, 1), (0, 3), (0, 4), (1, 2), (2, 3), (2, 4), (3, 4)]
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockT31, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.inception_s_1 = InceptionS(self.Cin, self.Cout, stride=self.stride, affine=self.affine)
        self.conv11 = ConvBNReLU(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.inception_s_3 = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        if self.stride == 2 or self.Cin != self.Cout:
            self.input_proj_3 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0,
                                       affine=self.affine)
            self.input_proj_4 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0,
                                       affine=self.affine)
        else:
            self.input_proj_3 = None
            self.input_proj_4 = None
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        if self.input_proj_3 is not None:
            in_3 = self.input_proj_3(inputs)
            in_4 = self.input_proj_4(inputs)
        else:
            in_3 = inputs
            in_4 = inputs
        out_1 = self.inception_s_1(inputs)
        out_2 = self.conv11(out_1)
        out_3 = self.inception_s_3(in_3 + out_2)
        output = self.relu(in_4 + out_2 + out_3)
        return output



class BlockSrc0(nn.Module):
    """
    [0, 2, 1, 1, 8],  # op_list,
    [[0, 1, 2, 1, 3],
     [1, 2, 3, 4, 4]]  # adj_list
    Conv op at inds: [1] can function as a reduction op
    70.83%/13.59M/2.87234ms
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockSrc0, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv11 = ConvBNReLU(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0, affine=self.affine)
        self.conv33_2 = ConvBNReLU(self.Cout, self.Cout, kernel_size=3, stride=1, padding=1, affine=self.affine)
        self.conv33_3 = ConvBN(self.Cout, self.Cout, kernel_size=3, stride=1, padding=1, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        inputs = self.conv11(inputs)
        out_2 = self.conv33_2(inputs)
        out_3 = self.conv33_3(out_2)
        output = self.relu(inputs + out_3)
        return output


class BlockSrc1(nn.Module):
    """
    [0, 2, 1, 1, 8],
    [[0, 0, 2, 1, 3],
     [1, 2, 3, 4, 4]]
    Conv op at inds: [1, 3] can function as a reduction op
    70.11%/9.72M/2.6289ms
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockSrc1, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv11 = ConvBNReLU(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0, affine=self.affine)
        self.conv33_2 = ConvBNReLU(self.Cin, self.Cin, kernel_size=3, stride=1, padding=1, affine=self.affine)
        self.conv33_3 = ConvBN(self.Cin, self.Cout, kernel_size=3, stride=self.stride, padding=1, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        shortcut = self.conv11(inputs)
        out_2 = self.conv33_2(inputs)
        out_3 = self.conv33_3(out_2)
        output = self.relu(shortcut + out_3)
        return output


class BlockSrc3(nn.Module):
    """
    [0, 2, 2, 1, 2, 8],
    [[0, 0, 2, 3, 4, 1],
     [1, 2, 3, 4, 5, 5]]
    Conv op at inds: [1, 3] can function as a reduction op
    68.96%/6.91M/2.3003ms
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockSrc3, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv11_1 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0, affine=self.affine)
        self.conv11_2 = ConvBNReLU(self.Cin, self.Cin, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.conv33 = ConvBNReLU(self.Cin, self.Cout, kernel_size=3, stride=self.stride, padding=1, affine=self.affine)
        self.conv11_4 = ConvBN(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        shortcut = self.conv11_1(inputs)
        out_2 = self.conv11_2(inputs)
        out_3 = self.conv33(out_2)
        out_4 = self.conv11_4(out_3)
        output = self.relu(shortcut + out_4)
        return output


class BlockSrc7(nn.Module):
    """
    [0, 2, 3, 3, 8],
    [[0, 1, 1, 2, 3],
     [1, 4, 2, 3, 4]]
    Conv op at inds: [1] can function as a reduction op
    70.89%/9.42M/2.9014ms
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockSrc7, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv11 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0, affine=self.affine)
        self.inception_s_1 = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        self.inception_s_2 = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        inputs = self.conv11(inputs)
        out_2 = self.inception_s_1(inputs)
        out_3 = self.inception_s_2(out_2)
        output = self.relu(inputs + out_3)
        return output


class BlockSrc8(nn.Module):
    """
    [0, 2, 3, 2, 8],
    [[0, 1, 1, 2, 3],
     [1, 4, 2, 3, 4]]
    Conv op at inds: [1] can function as a reduction op
    67.85%/5.94M/2.19756ms
    """
    def __init__(self, C_in, C_out, b_stride=1, b_affine=True):
        super(BlockSrc8, self).__init__()
        self.Cin = C_in
        self.Cout = C_out
        self.stride = b_stride
        self.affine = b_affine
        self.conv11_1 = ConvBN(self.Cin, self.Cout, kernel_size=1, stride=self.stride, padding=0, affine=self.affine)
        self.inception_s = InceptionS(self.Cout, self.Cout, stride=1, affine=self.affine)
        self.conv11_3 = ConvBN(self.Cout, self.Cout, kernel_size=1, stride=1, padding=0, affine=self.affine)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        inputs = self.conv11_1(inputs)
        out_2 = self.inception_s(inputs)
        out_3 = self.conv11_3(out_2)
        output = self.relu(inputs + out_3)
        return output


def benchmark_two_path_op_flops(C_in=32, W=32, H=32):
    from thop import profile
    from utils.model_utils import device
    from search_space.hiaml_two_path.constants import TWO_PATH_OP2IDX
    rand_input = torch.randn(1, C_in, W, H).to(device())
    pairs = []
    for op in TWO_PATH_OP2IDX.keys():
        if op == "input" or op == "output": continue
        C_out = C_in
        model = get_two_path_op_model(C_in, C_out, stride=1, op_name=op)
        model = model.to(device())
        out = model(rand_input)
        assert out.shape[1] == C_out
        macs, params = profile(model, inputs=(rand_input,))
        pairs.append((op, 2 * macs))
    pairs.sort(key=lambda t: t[1], reverse=True)
    print("{} ops benchmarked".format(len(pairs)))
    print(pairs)


def benchmark_hiaml_op_flops(C_in=32, W=32, H=32):
    from thop import profile
    from utils.model_utils import device
    from search_space.hiaml_two_path.constants import HIAML_OPS
    rand_input = torch.randn(1, C_in, W, H).to(device())
    pairs = []
    for op in HIAML_OPS:
        model = get_hiaml_block_op_model(C_in, 2 * C_in, stride=2, op_name=op)
        model = model.to(device())
        macs, params = profile(model, inputs=(rand_input,))
        pairs.append((op, 2 * macs))
    pairs.sort(key=lambda t: t[1], reverse=True)
    print(pairs)


if __name__ == "__main__":

    benchmark_two_path_op_flops()

    benchmark_hiaml_op_flops()

    print("done")
