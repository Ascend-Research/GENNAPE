import copy
import torch
import random
import collections
import numpy as np
from params import *
import torch.nn as nn
from tqdm import tqdm
from thop import profile
import torch.nn.functional as F
from collections import namedtuple
import torchvision.datasets as dset
from utils.eval_utils import accuracy
from utils.misc_utils import AverageMeter
from utils.model_utils import device, can_parallel, add_weight_decay


"""
DARTS models in a file
"""


DARTS_OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'scarlet'     : lambda C, stride, affine: nn.Conv2d(C, C, stride, stride=stride, bias=False),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
    'conv1x1': lambda C, stride, affine: NASBenchConv(C, C, 1, stride, padding=0, affine=affine),
    'conv3x3': lambda C, stride, affine: NASBenchConv(C, C, 3, stride, padding=1, affine=affine)
}


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


DARTS_PRIMITIVES = [
    # 'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


DARTS_SUPERNET_PRIMITIVES = [
    # 'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


DARTS_SUPERNET_OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: nn.Sequential(Identity(), nn.BatchNorm2d(C, affine=False))
                                                if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'scarlet'     : lambda C, stride, affine: nn.Conv2d(C, C, stride, stride=stride, bias=False),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
    'conv1x1': lambda C, stride, affine: NASBenchConv(C, C, 1, stride, padding=0, affine=affine),
    'conv3x3': lambda C, stride, affine: NASBenchConv(C, C, 3, stride, padding=1, affine=affine)
}


PRIMITIVES = DARTS_PRIMITIVES


class NASBenchConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding=0, affine=True):
        super(NASBenchConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """It's actually a depthwise dil conv"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = DARTS_OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, alphaRow):
        # Return weighted sum of all operations.
        return sum(alpha * op(x) for alpha, op in zip(alphaRow, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps): # Loop on nodes
            for j in range(2 + i): # Loop on edges
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0) # Either a factorizedReduce or ReLUConvBN (regularCNN)
        s1 = self.preprocess1(s1) # s0 and s1 initial shape

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            # In the beginning, offset = 0, i = 0, j = 0 input is state[0]
            # So alphas row 0 -> edge between (0, 2), input node 0, internal node 2
            # Next offset = 0, i = 0, j = 1 input is state[1]
            # So alphas row 1 -> edge between (1, 2), input node 1, internal node 2
            # Next offset = 2, i = 1, j = 0 input is state[0]
            # So alphas row 2 -> edge between (0, 3), input node 0, internal node 3
            # Next offset = 2, i = 1, j = 1 input is state[1]
            # So alphas row 3 -> edge between (1, 3), input node 1, internal node 3
            # Next offset = 2, i = 1, j = 2 input is state[2]
            # So alphas row 4 -> edge between (2, 3), internal node 2, internal node 3
            # ...
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, stemDict, steps=4, multiplier=4, stem_multiplier=3,
                 softmax=True, skipNones=False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._stemDict = stemDict
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._oneshotMode = True
        self._softmax = softmax
        self._skipNones = skipNones

        C_curr = stem_multiplier * C  # 3 * 16 channels in CNN.
        self.stem = nn.Sequential(
            nn.Conv2d(stemDict['channels'], C_curr, stemDict['filter'], padding=stemDict['padding'], bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # C_prev_prev and C_prev is output channel size, but
        # C_curr is input channel size.
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # every layer has a cell, cells are either normal or reduction cells
        # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
        for i in range(layers):  # layers = 8
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            # each cell takes as an input the output of two previous cells
            # and whether it is a reduction cell or proceeded by one.
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._stemDict,
                            self._steps, self._multiplier, self._stem_multiplier,
                            self._softmax, self._skipNones).to(device())
        model_new.setOneshot(False)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self._oneshotMode:
                weights = self.oneshotAlphas
            else:
                if cell.reduction:
                    weights = self.alphas_reduce
                else:
                    weights = self.alphas_normal
                if self._softmax:
                    weights = F.softmax(weights, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    # Weights for linear combination. For normal and reduce.
    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.oneshotAlphas = torch.ones(k, num_ops)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def setAlphas(self, norm, redu):
        self.alphas_normal = nn.Parameter(norm)
        self.alphas_reduce = nn.Parameter(redu)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def getOneshot(self):
        return self._oneshotMode

    def setOneshot(self, boolVal):
        if isinstance(boolVal, bool):
            self._oneshotMode = boolVal

    def arch_parameters(self):
        return self._arch_parameters

    # Very complicated
    # Edges are operations, nodes are representations.
    # A way to extract the graph figure from the weights.
    # When is it called? After each validation run. Prints to log.
    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                if self._skipNones:
                    W[:, 0] = -100
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


def infer(valid_queue, model, criterion, report_freq=50, log_f=print):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    cummulativeLoss = 0

    for step, (X, y) in enumerate(valid_queue):
        X = X.to(device())
        y = y.to(device())

        logits = model(X)
        loss = criterion(logits, y)
        cummulativeLoss += loss.item()

        prec1, prec5 = accuracy(logits, y, topk=(1, 5))
        n = X.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0 and log_f is not None:
            log_f('valid %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))

    model.train()
    return top1.avg, objs.avg, cummulativeLoss


def load(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device()), strict=False)


def sample_alphas(n_steps=4):
    k = sum(1 for i in range(n_steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)
    alphas_normal = torch.randn(k, num_ops)
    alphas_reduce = torch.randn(k, num_ops)
    alphas_normal = F.softmax(alphas_normal, dim=-1).cpu()
    alphas_reduce = F.softmax(alphas_reduce, dim=-1).cpu()
    return alphas_normal, alphas_reduce


def alphas_to_genotype(n_steps, alphas_normal, alphas_reduce, multiplier=4):

    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(n_steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene

    # gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
    # gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

    gene_normal = _parse(alphas_normal.data.cpu().numpy())
    gene_reduce = _parse(alphas_reduce.data.cpu().numpy())

    concat = range(2 + n_steps - multiplier, n_steps + 2)
    genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)
    return genotype


def discretize_alpha(alpha):
    new_alpha = torch.zeros_like(alpha)
    n = 2
    start = 0
    for i in range(4):
        end = start + n
        W = alpha[start:end]
        edges = sorted(range(i + 2),
                       key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
            new_alpha[start+j, k_best] = 1
        start = end
        n += 1
    return new_alpha


def sample_genotype(n_steps=4):
    alphas_normal, alphas_reduce = sample_alphas(n_steps)
    return alphas_to_genotype(n_steps, alphas_reduce, alphas_reduce)


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).to(device()).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def getStemParams(dataset='CIFAR'):
    stemDict = {}
    if 'CIFAR' in dataset:
        stemDict['channels'] = 3
        stemDict['filter'] = 3
        stemDict['padding'] = 1
        return stemDict

    return NotImplementedError


class EvalCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(EvalCell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = DARTS_OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class EvalNetwork(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, stemDict, stem_multiplier=3):
        super(EvalNetwork, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(stemDict['channels'], C_curr, stemDict['filter'], padding=stemDict['padding'], bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = EvalCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


def full_train_darts(genotype, init_channels=36, classes=10, layers=20, auxiliary=True, auxiliary_weight=0.4,
                     grad_clip=5.0, report_freq=50, learning_rate=0.025, momentum=0.9, weight_decay=3e-4,
                     epochs=600, drop_path_prob=0.2, batch_size=64, cutout=True, log_f=print, num_workers=0):
    log_f("Full train genotype: {}".format(genotype))
    classes, train_loader, test_loader = trainTestSet(data="CIFAR10" if classes == 10 else "CIFAR100",
                                                      cutout=cutout, batch_size=batch_size, num_workers=num_workers)
    model = EvalNetwork(init_channels, classes, layers, auxiliary, genotype, getStemParams()).to(device())
    model.drop_path_prob = 0
    _input = torch.randn(1, 3, 32, 32).to(device())
    macs, params = profile(model, inputs=(_input,), verbose=False)
    log_f("FLOPS: %d; Params: %d" % (macs * 2, params))
    criterion = nn.CrossEntropyLoss().to(device())
    _p = add_weight_decay(model, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    bestDict = {'Best Test Acc': 0, 'Best Training Acc': 0}

    for epoch in range(epochs):
        log_f('epoch {} lr {}'.format(epoch, scheduler.get_lr()[0]))
        model.drop_path_prob = drop_path_prob * epoch / epochs

        train_acc, train_obj = _run_full_train_epoch(model, train_loader, optimizer, criterion,
                                                     auxiliary, auxiliary_weight, grad_clip, report_freq,
                                                     log_f=log_f)
        log_f('train_acc {}'.format(train_acc))

        if train_acc > bestDict['Best Training Acc']:
            bestDict['Best Training Acc'] = train_acc
            bestDict['Best Training Epoch'] = 0

        with torch.no_grad():
            test_acc, test_obj, _ = _full_train_infer(test_loader, model, criterion, report_freq, log_f=log_f)
        log_f('test_acc {}'.format(test_acc))

        if test_acc > bestDict['Best Test Acc']:
            bestDict['Best Test Acc'] = test_acc
            bestDict['Best Test Epoch'] = epoch

        # checkpoint
        torch.save(model.state_dict(), P_SEP.join([SAVED_MODELS_DIR, 'darts_full_train.pt']))
        scheduler.step()

    for key in sorted(bestDict.keys()):
        newLine = " "
        if "Alpha" in key:
            newLine = "\n"
        log_f("{}:{}{}".format(key, newLine, bestDict[key]))


def _full_train_infer(valid_queue, model, criterion, report_freq=50, log_f=print):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    cummulativeLoss = 0

    for step, (X, y) in enumerate(valid_queue):
        X = X.to(device())
        y = y.to(device())

        logits, _ = model(X)
        loss = criterion(logits, y)
        cummulativeLoss += loss.item()

        prec1, prec5 = accuracy(logits, y, topk=(1, 5))
        n = X.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0 and log_f is not None:
            log_f('valid %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))

    model.train()
    return top1.avg, objs.avg, cummulativeLoss


def _run_full_train_epoch(model, train_queue, optimizer, criterion, auxiliary, auxiliary_weight, grad_clip,
                          report_freq, log_f=print):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.to(device())
        target = target.to(device())

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0:
            log_f('train {} {:.3f} {:.3f} {:.3f}'.format(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


def searchSet(data="CIFAR10", cutout=True, debug=False, train_portion=0.5, batch_size=256,
              num_workers=0):

    classes, trainTransform, _ = _getTransforms(data, cutout)
    trainSet = _getDset(data, trainTransform)

    numTrain = len(trainSet)
    if debug: numTrain = int(numTrain / 50)
    indices = list(range(numTrain))
    split = int(np.floor(train_portion * numTrain))

    trainQueue = torch.utils.data.DataLoader(
        trainSet, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=num_workers)

    validQueue = torch.utils.data.DataLoader(
        trainSet, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:numTrain]),
        pin_memory=True, num_workers=num_workers)

    return classes, trainQueue, validQueue


def trainTestSet(data="CIFAR10", cutout=True, debug=False, batch_size=100, num_workers=0):

    classes, trainTransform, validTransform = _getTransforms(data, cutout)
    trainSet, testSet = _getDset(data, trainTransform, test=True, vTransform=validTransform)

    if debug:
        numTrain = int(len(trainSet) / 50)
        numTest = int(len(testSet) / 50)
        trainIndices = list(range(numTrain))
        testIndices = list(range(numTest))
        trainSampler = torch.utils.data.sampler.SubsetRandomSampler(trainIndices)
        testSampler = torch.utils.data.sampler.SubsetRandomSampler(testIndices)
        tShuffle = False

    else:
        trainSampler = None
        tShuffle = True
        testSampler = None

    trainQueue = torch.utils.data.DataLoader(
        trainSet, batch_size=batch_size, shuffle=tShuffle, pin_memory=True, num_workers=num_workers,
        sampler=trainSampler)

    testQueue = torch.utils.data.DataLoader(
        testSet, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
        sampler=testSampler)

    return classes, trainQueue, testQueue


def _getTransforms(data, cutout):

    if 'CIFAR' in data:
        print("Data is CIFAR")
        return data_transforms_cifar(data, cutout)

    else:
        raise NotImplementedError


def _getDset(data, tTransform, test=False, vTransform=None):

    if '100' in data:
        print("Fetching CIFAR100")
        trainSet = dset.CIFAR100(root=DATA_DIR, train=True, download=True, transform=tTransform)
        if test:
            testSet = dset.CIFAR100(root=DATA_DIR, train=False, download=True, transform=vTransform)
            return trainSet, testSet
        return trainSet

    elif '10' in data:
        print("Fetching CIFAR10")
        trainSet = dset.CIFAR10(root=DATA_DIR, train=True, download=True, transform=tTransform)
        if test:
            testSet = dset.CIFAR10(root=DATA_DIR, train=False, download=True, transform=vTransform)
            return trainSet, testSet
        return trainSet

    else:
        raise NotImplementedError


def data_transforms_cifar(name, cutout=False):
    import torchvision.transforms as transforms
    if '100' in name:
        print("Transform for CIFAR100")
        CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
        CIFAR_STD = [0.2675, 0.2565, 0.2761]
        CIFAR_CO = 8
        CIFAR_CLASS = 100

    else:
        print("Transform for CIFAR10")
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        CIFAR_CO = 16
        CIFAR_CLASS = 10

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(CIFAR_CO))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return CIFAR_CLASS, train_transform, valid_transform


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class EvalCellV2(nn.Module):
    """
    Not constrained by the each node can have 2 edges going into it constraint
    """
    def __init__(self, op_names, ni2de_map, concat, n_internal_nodes,
                 C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(EvalCellV2, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self.ops = nn.ModuleList()
        self.n_internal_nodes = n_internal_nodes
        self.concat = concat
        self.multiplier = len(concat)

        # Maps dst node idx to a list of tuples
        # Each tuple is a pair of (node_idx, input_state(node)_idx)
        # Basically to get the current node, we input the designated state to each op and sum the results
        self.node_input_map = collections.defaultdict(list)
        for ni, name in enumerate(op_names):
            src_n, dst_n = ni2de_map[ni + 2] # Must add 2 here to account for input/output nodes
            # If this op connects to one of the output node, i.e. src_n is 0 or 1
            # Then it might function as a reduction op
            stride = 2 if reduction and src_n < 2 else 1
            op = DARTS_OPS[name](C, stride, True)
            self.ops.append(op)
            self.node_input_map[dst_n].append((ni, src_n))

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self.n_internal_nodes):
            dst_node_idx = 2 + i
            tmp_outputs = []
            for op_idx, src_idx in self.node_input_map[dst_node_idx]:
                input_node = states[src_idx]
                op = self.ops[op_idx]
                h = op(input_node)
                if self.training and drop_prob > 0.:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_prob)
                tmp_outputs.append(h)
            states.append(sum(tmp_outputs))
        return torch.cat([states[i] for i in self.concat], dim=1)


class EvalNetworkV2(nn.Module):

    def __init__(self, normal_op_names, reduce_op_names, ni2de_map, concat, n_internal_nodes,
                 C, num_classes, layers, auxiliary, stemDict, stem_multiplier=3):
        super(EvalNetworkV2, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(stemDict['channels'], C_curr, stemDict['filter'], padding=stemDict['padding'], bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = EvalCellV2(reduce_op_names, ni2de_map, concat, n_internal_nodes,
                                  C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            else:
                reduction = False
                cell = EvalCellV2(normal_op_names, ni2de_map, concat, n_internal_nodes,
                                  C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


def full_train_darts_v2(normal_op_names, reduce_op_names, ni2de_map, concat, n_internal_nodes,
                        n_classes, train_loader, dev_loader, test_loader,
                        init_channels=36, layers=20, auxiliary=True, auxiliary_weight=0.4,
                        grad_clip=5.0, report_freq=50, learning_rate=0.025, momentum=0.9, weight_decay=3e-4,
                        epochs=600, drop_path_prob=0.2, allow_parallel=True,
                        log_f=print):
    log_f("Full train for normal ops: {}, reduce ops: {}".format(normal_op_names, reduce_op_names))
    model = EvalNetworkV2(normal_op_names, reduce_op_names, ni2de_map, concat, n_internal_nodes,
                          init_channels, n_classes, layers, auxiliary, getStemParams()).to(device())
    model.drop_path_prob = 0
    _input = torch.randn(1, 3, 32, 32).to(device())
    macs, params = profile(model, inputs=(_input,), verbose=False)
    log_f("FLOPS: %d; Params: %d" % (macs * 2, params))
    criterion = nn.CrossEntropyLoss().to(device())
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    bestDict = {'Best Dev Acc': 0, 'Best Training Acc': 0}
    best_model = None
    for epoch in range(epochs):
        log_f('epoch {} lr {}'.format(epoch, scheduler.get_lr()[0]))
        model.drop_path_prob = drop_path_prob * epoch / epochs
        if allow_parallel and can_parallel():
            model = nn.DataParallel(model).to(device())
        train_acc, train_obj = _run_full_train_epoch(model, train_loader, optimizer, criterion,
                                                     auxiliary, auxiliary_weight, grad_clip, report_freq,
                                                     log_f=log_f)
        if isinstance(model, nn.DataParallel):
            model = model.module
        log_f('train_acc {}'.format(train_acc))

        if train_acc > bestDict['Best Training Acc']:
            bestDict['Best Training Acc'] = train_acc
            bestDict['Best Training Epoch'] = epoch

        with torch.no_grad():
            dev_acc, dev_obj, _ = _full_train_infer(dev_loader, model, criterion, report_freq, log_f=log_f)
        log_f('dev_acc {}'.format(dev_acc))

        if dev_acc > bestDict['Best Dev Acc']:
            bestDict['Best Dev Acc'] = dev_acc
            bestDict['Best Dev Epoch'] = epoch
            best_model = copy.deepcopy(model)

        # checkpoint
        torch.save(model.state_dict(), P_SEP.join([SAVED_MODELS_DIR, 'darts_full_train.pt']))
        scheduler.step()

    for key in sorted(bestDict.keys()):
        newLine = " "
        if "Alpha" in key:
            newLine = "\n"
        log_f("{}:{}{}".format(key, newLine, bestDict[key]))

    with torch.no_grad():
        test_acc, _, _ = _full_train_infer(test_loader, best_model, criterion, report_freq, log_f=log_f)
    log_f('Test acc using best model: {}'.format(test_acc))
    return test_acc


class MixedOpV2(nn.Module):
    """Slightly modified"""
    def __init__(self, C, stride):
        super(MixedOpV2, self).__init__()
        self.op_label2idx = {}
        self._ops = nn.ModuleList()
        for idx, primitive in enumerate(DARTS_SUPERNET_PRIMITIVES):
            op = DARTS_OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self.op_label2idx[primitive] = idx

    def forward(self, x, alphaRow):
        # Return weighted sum of all operations.
        return sum(alpha * op(x) for alpha, op in zip(alphaRow, self._ops))


class SuperCellV2(nn.Module):
    """Modified to be more like the EvalCell"""
    def __init__(self, n_ops_per_cell, ni2de_map, concat, n_internal_nodes,
                 C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(SuperCellV2, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        self.ops = nn.ModuleList()
        self.n_internal_nodes = n_internal_nodes
        self.concat = concat
        self.multiplier = len(concat)
        self.op_label2idx = None
        self.reduction = reduction
        # Maps dst node idx to a list of tuples
        # Each tuple is a pair of (node_idx, input_state(node)_idx)
        # Basically to get the current node, we input the designated state to each op and sum the results
        self.node_input_map = collections.defaultdict(list)
        for ni in range(n_ops_per_cell):
            src_n, dst_n = ni2de_map[ni + 2] # Must add 2 here to account for input/output nodes
            # If this op connects to one of the output node, i.e. src_n is 0 or 1
            # Then it might function as a reduction op
            stride = 2 if reduction and src_n < 2 else 1
            op = MixedOpV2(C, stride)
            self.ops.append(op)
            self.op_label2idx = op.op_label2idx
            self.node_input_map[dst_n].append((ni, src_n))

    def forward(self, s0, s1, b_alphas):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        curr_op_idx = 0
        for i in range(self.n_internal_nodes):
            dst_node_idx = 2 + i
            tmp_outputs = []
            for op_idx, src_idx in self.node_input_map[dst_node_idx]:
                input_node = states[src_idx]
                op = self.ops[op_idx]
                h = op(input_node, b_alphas[curr_op_idx])
                curr_op_idx += 1
                tmp_outputs.append(h)
            states.append(sum(tmp_outputs))
        return torch.cat([states[i] for i in self.concat], dim=1)


class SuperNetV2(nn.Module):
    """Modified to work with SuperCellV2"""
    def __init__(self, n_ops_per_cell, ni2de_map, concat, n_internal_nodes, C, num_classes, layers, stemDict,
                 stem_multiplier=3):
        super(SuperNetV2, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._stemDict = stemDict
        self._stem_multiplier = stem_multiplier
        self.n_ops_per_cell = n_ops_per_cell
        self.ni2de_map = ni2de_map
        self.concat = concat
        self.n_internal_nodes = n_internal_nodes
        self.op_label2idx = None
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(stemDict['channels'], C_curr, stemDict['filter'], padding=stemDict['padding'], bias=False),
            nn.BatchNorm2d(C_curr)
        )
        # C_prev_prev and C_prev is output channel size, but
        # C_curr is input channel size.
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        # every layer has a cell, cells are either normal or reduction cells
        # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
        for i in range(layers):  # layers = 8
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            # each cell takes as an input the output of two previous cells
            # and whether it is a reduction cell or proceeded by one.
            cell = SuperCellV2(n_ops_per_cell, ni2de_map, concat, n_internal_nodes,
                               C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.op_label2idx = cell.op_label2idx
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, len(concat) * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x, b_alphas_normal, b_alphas_reduce):
        assert len(b_alphas_reduce) == len(b_alphas_normal) == self.n_ops_per_cell
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                b_alphas = b_alphas_reduce
            else:
                b_alphas = b_alphas_normal
            s0, s1 = s1, cell(s0, s1, b_alphas)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


def make_darts_c10_supernet_v2(ni2de_map, concat, stemDict, init_C=16, layers=20, n_internal_nodes=4, num_classes=10,
                               n_ops_per_cell=14, stem_multiplier=3):
    model = SuperNetV2(n_ops_per_cell, ni2de_map, concat, n_internal_nodes, init_C, num_classes, layers, stemDict,
                       stem_multiplier=stem_multiplier)
    return model


def uniform_sample_b_alphas(n_edges_per_cell, n_candidate_ops):
    b_alphas = []
    for ei in range(n_edges_per_cell):
        op_idx = random.randint(0, n_candidate_ops-1)
        onehot_vec = torch.zeros(n_candidate_ops).to(device())
        onehot_vec[op_idx] = 1
        b_alphas.append(onehot_vec)
    return b_alphas


def train_darts_supernet_v2(num_epochs, model, train_loader, criterion, optimizer, book_keeper,
                            max_grad_norm=5.0, completed_epochs=0, epoch_lr_scheduler=None,
                            dev_loader=None, allow_parallel=False, checkpoint=True):
    model = model.to(device())
    criterion = criterion.to(device())
    unique_archs = set()
    for epoch in range(num_epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        if allow_parallel and can_parallel():
            model = torch.nn.DataParallel(model)
        run_darts_supernet_v2_epoch(report_epoch, model, train_loader, criterion,
                                    optimizer=optimizer, log_f=book_keeper.log,
                                    desc="Train", max_grad_norm=max_grad_norm,
                                    global_unique_archs=unique_archs)
        if epoch_lr_scheduler is not None:
            epoch_lr_scheduler.step()
            book_keeper.log("New lr: {}".format(epoch_lr_scheduler.get_last_lr()))
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        book_keeper.log("Trained on {} unique archs so far".format(len(unique_archs)))

        if checkpoint:
            book_keeper.checkpoint_model("_latest.pt", report_epoch, model, optimizer)

        if dev_loader is not None:
            _, dev_acc = run_darts_supernet_v2_epoch(report_epoch, model, dev_loader, criterion,
                                                     log_f=book_keeper.log, desc="Dev")
            if checkpoint:
                book_keeper.checkpoint_model("_best.pt", report_epoch, model, optimizer, eval_perf=dev_acc)
                book_keeper.report_curr_best()
        book_keeper.log("")


def run_darts_supernet_v2_epoch(i, model, loader, criterion,
                                n_edges_per_cell=14, n_candidate_ops=len(DARTS_SUPERNET_PRIMITIVES),
                                optimizer=None, log_f=print, desc="Train",
                                max_grad_norm=5.0, global_unique_archs=None):
    avg_loss = AverageMeter()
    avg_top_1_acc = AverageMeter()
    avg_top_5_acc = AverageMeter()
    unique_archs = set()
    for feature, target in tqdm(loader, desc=desc, ascii=True):
        feature = feature.to(device())
        target = target.to(device())
        b_alphas_normal = uniform_sample_b_alphas(n_edges_per_cell, n_candidate_ops)
        b_alphas_reduce = uniform_sample_b_alphas(n_edges_per_cell, n_candidate_ops)
        logits = model(feature, b_alphas_normal, b_alphas_reduce)
        loss = criterion(logits, target)
        if torch.isnan(loss).any():
            continue
        unique_archs.add("|".join([str(b_alphas_normal), str(b_alphas_reduce)]))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        p1, p5 = accuracy(logits, target, topk=(1, 5))
        avg_loss.update(loss.item(), feature.shape[0])
        avg_top_1_acc.update(p1.item(), feature.shape[0])
        avg_top_5_acc.update(p5.item(), feature.shape[0])
    msg = "{} epoch {} loss: {}, top-1 acc: {}, top-5 acc: {}".format(desc, i, avg_loss.avg,
                                                                      avg_top_1_acc.avg, avg_top_5_acc.avg)
    if global_unique_archs is not None:
        global_unique_archs.update(unique_archs)
    if log_f is not None:
        log_f(msg)
        log_f("Num unique arch sampled: {}".format(len(unique_archs)))
    return avg_loss.avg, avg_top_1_acc.avg

