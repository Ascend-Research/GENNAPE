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


def get_op_model(C_in, C_out, stride, op_name,
                 bn, bias):
    if op_name.startswith("conv"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k, padding = _decode_kernel_padding_size(conv_op_str.replace("conv", ""))
        if len(parts) == 1:
            op = ConvBNReLU(C_in, C_out, kernel_size=k, stride=stride,
                            padding=padding, bn=bn, bias=bias)
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = ConvBNReLU(C_hidden, C_hidden, kernel_size=k, stride=stride,
                                      padding=padding, bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
            elif opt.startswith("e"):
                e = _decode_int_value(opt, "e")
                C_hidden = int(C_in * e)
                inner_op = ConvBNReLU(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0], bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
            else:
                raise ValueError
        else:
            raise ValueError

    elif op_name.startswith("depthwise"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k, padding = _decode_kernel_padding_size(conv_op_str.replace("depthwise", ""))
        if len(parts) == 1:
            op = DepthwiseBNReLU(C_in, C_out, kernel_size=k, stride=stride,
                                 padding=padding, bn=bn, bias=bias)
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = DepthwiseBNReLU(C_hidden, C_hidden, kernel_size=k, stride=stride,
                                           padding=padding, bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
            elif opt.startswith("e"):
                e = _decode_int_value(opt, "e")
                C_hidden = int(C_in * e)
                inner_op = DepthwiseBNReLU(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                           padding=padding[0], bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
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
                            padding=padding[0], bn=bn, bias=bias)
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = InceptionS(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0], bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
            elif opt.startswith("e"):
                e = _decode_int_value(opt, "e")
                C_hidden = int(C_in * e)
                inner_op = InceptionS(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0], bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
            else:
                raise ValueError
        else:
            raise ValueError

    elif op_name.startswith("inception_p"):
        parts = op_name.split("-")
        conv_op_str = parts[0]
        k, padding = _decode_kernel_padding_size(conv_op_str.replace("inception_p", ""))
        if len(parts) == 1:
            op = InceptionP(C_in, C_out, kernel_size=k[0], stride=stride,
                            padding=padding[0], bn=bn, bias=bias)
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = InceptionP(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0], bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
            elif opt.startswith("e"):
                e = _decode_int_value(opt, "e")
                C_hidden = int(C_in * e)
                inner_op = InceptionP(C_hidden, C_hidden, kernel_size=k[0], stride=stride,
                                      padding=padding[0],
                                      bn=bn, bias=bias)
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
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
                ConvBNReLU(C_in, C_out, kernel_size=k, stride=1,
                           padding=padding, bn=bn, bias=bias),
                ConvBNReLU(C_out, C_out, kernel_size=k, stride=stride,
                           padding=padding, bn=bn, bias=bias),
            )
        elif len(parts) == 2:
            _, opt = parts
            if opt.startswith("r"):
                r = _decode_int_value(opt, "r")
                C_hidden = max(1, C_in // r)
                inner_op = nn.Sequential(
                    ConvBNReLU(C_hidden, C_hidden, kernel_size=k, stride=stride,
                               padding=padding, bn=bn, bias=bias),
                    ConvBNReLU(C_hidden, C_hidden, kernel_size=k, stride=1,
                               padding=padding, bn=bn, bias=bias),
                )
                op = ChannelResizeWrapper(C_in, C_hidden, C_out, inner_op,
                                          bn=bn)
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
            op = Bottleneck(C_in, C_hidden, C_out, kernel_size=k, stride=stride, padding=k//2,
                            bn=bn, bias=bias)
        else:
            raise ValueError

    elif op_name.startswith("avgpool"):
        op = AvgPool(C_in, C_out, kernel_size=3, stride=stride, padding=1, bn=bn)

    elif op_name.startswith("maxpool"):
        op = MaxPool(C_in, C_out, kernel_size=3, stride=stride, padding=1, bn=bn)

    else:
        raise ValueError("Unknown op name: {}".format(op_name))
    return op


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 affine=True, bn=True, bias=True):
        super(ConvBNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(C_out, affine=affine) if bn else Identity(),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.op(x)


class ConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 affine=True, bn=True, bias=True):
        super(ConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(C_out, affine=affine) if bn else Identity(),
        )

    def forward(self, x):
        return self.op(x)


class DepthwiseBNReLU(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 affine=True, bn=True, bias=True):
        super(DepthwiseBNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False) if C_in != C_out else Identity(),
            nn.Conv2d(C_out, C_out,
                      groups=C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(C_out, affine=affine) if bn else Identity(),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.op(x)


class ChannelResizeWrapper(nn.Module):

    def __init__(self, C_in, C_hidden, C_out, op,
                 bn=True):
        super(ChannelResizeWrapper, self).__init__()
        self.proj1 = ConvBN(C_in, C_hidden, 1, 1, 0,
                            bn=bn, bias=False)
        self.op = op
        self.proj2 = ConvBN(C_hidden, C_out, 1, 1, 0,
                            bn=bn, bias=False)

    def forward(self, x):
        return self.proj2(self.op(self.proj1(x)))


class Bottleneck(nn.Module):

    def __init__(self, C_in, C_hidden, C_out, kernel_size, stride, padding,
                 affine=True, bn=True, bias=True):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBNReLU(C_in, C_hidden, kernel_size, stride, padding,
                                affine=affine, bn=bn, bias=bias)
        self.conv2 = ConvBNReLU(C_hidden, C_out, kernel_size, 1, padding,
                                affine=affine, bn=bn, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class InceptionS(nn.Module):

    def __init__(self, Cin, Cout, kernel_size=3, stride=1, padding=1,
                 affine=True, bn=True, bias=True):
        super(InceptionS, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(Cin, Cout, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding), bias=bias),
            nn.BatchNorm2d(Cout, affine=affine) if bn else Identity(),
            nn.ReLU6(),
            nn.Conv2d(Cout, Cout, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0), bias=bias),
            nn.BatchNorm2d(Cout, affine=affine) if bn else Identity(),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.op(x)


class InceptionP(nn.Module):

    def __init__(self, Cin, Cout, kernel_size=3, stride=1, padding=1,
                 affine=True, bn=True, bias=True):
        super(InceptionP, self).__init__()
        C1 = Cout // 2
        if stride == 1:
            self.op1 = ConvBN(Cin, C1, kernel_size=(1, kernel_size), stride=1, padding=(0, padding),
                              affine=affine, bn=bn, bias=bias)
            self.op2 = ConvBN(Cin, Cout - C1, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0),
                              affine=affine, bn=bn, bias=bias)
        else:
            self.op1 = nn.Sequential(
                ConvBN(Cin, C1, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding),
                       affine=affine, bn=bn, bias=bias),
                ConvBN(C1, C1, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0),
                       affine=affine, bn=bn, bias=bias),
            )
            self.op2 = nn.Sequential(
                ConvBN(Cin, Cout - C1, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0),
                       affine=affine, bn=bn, bias=bias),
                ConvBN(Cout - C1, Cout - C1, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding),
                       affine=affine, bn=bn, bias=bias),
            )
        self.activ = nn.ReLU6()

    def forward(self, x):
        x1 = self.activ(self.op1(x))
        x2 = self.activ(self.op2(x))
        return  self.activ(torch.cat([x1, x2], dim=1))


class AvgPool(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 affine=True, bn=True):
        super(AvgPool, self).__init__()
        assert C_in == C_out
        self.op = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride, padding),
            nn.BatchNorm2d(C_out, affine=affine) if bn else Identity(),
        )

    def forward(self, x):
        return self.op(x)


class MaxPool(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 affine=True, bn=True):
        super(MaxPool, self).__init__()
        assert C_in == C_out
        self.op = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride, padding),
            nn.BatchNorm2d(C_out, affine=affine) if bn else Identity(),
        )

    def forward(self, x):
        return self.op(x)


def benchmark_op_flops(C_in=32, W=32, H=32):
    from thop import profile
    from utils.model_utils import device
    from search_space.inception.constants import OP2IDX, CONV_OPS
    rand_input = torch.randn(1, C_in, W, H).to(device())
    pairs = []
    for op in OP2IDX.keys():
        if op == "input" or op == "output": continue
        C_out = C_in * 2 if op in CONV_OPS else C_in
        stride = 2 if op in CONV_OPS else 1
        model = get_op_model(C_in, C_out, stride=stride, op_name=op, bn=True, bias=True)
        model = model.to(device())
        out = model(rand_input)
        assert out.shape[1] == C_out
        macs, params = profile(model, inputs=(rand_input,))
        pairs.append((op, 2 * macs))
    pairs.sort(key=lambda t: t[1], reverse=True)
    print("{} ops benchmarked".format(len(pairs)))
    print(pairs)


if __name__ == "__main__":

    benchmark_op_flops()

    print("done")