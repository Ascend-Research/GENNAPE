"""
Defines constants for this search space
A short label for this space is: GPT
Main purpose of the space is to label test networks for generalizable predictors
"""

"""
Two path sub-space
"""
TWO_PATH_MIN_N_OPS_PER_BLOCK = 1
TWO_PATH_MAX_N_OPS_PER_BLOCK = 3

TWO_PATH_MIN_N_BLOCKS_PER_PATH = 2
TWO_PATH_MAX_N_BLOCKS_PER_PATH = 4

TWO_PATH_NUM_REDUCTIONS_PER_PATH = 2

assert TWO_PATH_MIN_N_BLOCKS_PER_PATH >= TWO_PATH_NUM_REDUCTIONS_PER_PATH

TWO_PATH_CONV_OPS = (
    "conv3x3",
    "double3x3",
    "conv5x5",
    "bottleneck-r2-k3",
    "bottleneck-r4-k3",
    "bottleneck-e2-k3",
    "inception_s3x3",
    "inception_s5x5",
)

TWO_PATH_SUPP_OPS = (
    "conv1x1",
    "bottleneck-r2-k1",
    "bottleneck-e2-k1",
    "avgpool3x3",
    "maxpool3x3",
)

TWO_PATH_OP2IDX = {}
for __op in TWO_PATH_CONV_OPS + TWO_PATH_SUPP_OPS:
    TWO_PATH_OP2IDX[__op] = len(TWO_PATH_OP2IDX)
TWO_PATH_IDX2OP = {v: k for k, v in TWO_PATH_OP2IDX.items()}
assert len(TWO_PATH_OP2IDX) == len(TWO_PATH_IDX2OP)


"""
HiAML sub-space
"""

HIAML_NUM_BLOCKS_PER_STACK = 2

HIAML_OPS = (
    "hiaml_z",
    "hiaml_a22",
    "hiaml_b23",
    "hiaml_j30",
    "hiaml_g30",
    "hiaml_r30",
    "hiaml_l20",
    "hiaml_t20",
    "hiaml_t31",
    "hiaml_src0",
    "hiaml_src1",
    "hiaml_src3",
    "hiaml_src7",
    "hiaml_src8",
)

HIAML_OP2IDX = {}
for __op in HIAML_OPS:
    HIAML_OP2IDX[__op] = len(HIAML_OP2IDX)
HIAML_IDX2OP = {v: k for k, v in HIAML_OP2IDX.items()}
assert len(HIAML_OP2IDX) == len(HIAML_IDX2OP)
