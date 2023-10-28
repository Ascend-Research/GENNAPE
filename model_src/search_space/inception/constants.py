

C_INIT = 32


N_STAGES = 4
REDUCTION_STAGE_INDS = (1, 2, 3)


MIN_CHANNEL_SIZE = 16


MIN_BLOCKS_PER_STAGE = 2
MAX_BLOCKS_PER_STAGE = 6


PATH_C_CHANGE_PROBS = (("reduce2", 0.3), ("no_change", 0.6), ("expand2", 0.1))


NUM_PATHS_IN_BLOCK = (1, 2, 3, 4)
# N_PATHS_TO_MAX_BLOCK_OPS = {
#     1 : 4,
#     2 : 8,
#     3 : 12,
#     4 : 16,
# }
N_PATHS_TO_MAX_BLOCK_OPS = {
    1 : 3,
    2 : 4,
    3 : 8,
    4 : 10,
}
N_PATHS_TO_MIN_N_OPS_PER_PATH = {
    1 : 1,
    2 : 1,
    3 : 2,
    4 : 2,
}


CONV_OPS = (
    "conv3x3",
    "inception_s3x3",
    "inception_p3x3",
    "depthwise3x3",
    "conv5x5",
    "depthwise5x5",
    "inception_s5x5",
    "inception_p5x5",
)


SUPP_OPS = (
    "conv1x1",
    "avgpool3x3",
    "maxpool3x3",
    "double1x1",
)

SUPP_C_CHANGE_CAPABLE_OPS = (
    "conv1x1",
    "double1x1",
)


OP2IDX = {
    # "input": 0,
    # "output": 1,
}
for __op in CONV_OPS + SUPP_OPS:
    OP2IDX[__op] = len(OP2IDX)
IDX2OP = {v: k for k, v in OP2IDX.items()}
assert len(OP2IDX) == len(IDX2OP)
