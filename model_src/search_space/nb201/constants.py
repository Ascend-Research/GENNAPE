"""
Defines constants for this search space
"""

OPS = (
    "nor_conv_3x3",
    "nor_conv_1x1",
    "none",
    "skip_connect",
    "avg_pool_3x3",
)


# Node(op) idx to DARTS edge (src node - dst node) map
NI2DE_MAP = {1: (0, 1), 2: (0, 2), 3: (1, 2), 4: (0, 3), 5: (1, 3), 6: (2, 3)}


# Captures in a node-based topo, what are the input inds to a certain op
# Assumes the first index is the input state
OP_INPUT_INDS = (
    (0, ), # First op takes the input, i.e., index 0
    (0, ), # Second op corresponds to the DARTS edge between the second state and input state
    (1, ), # Third op corresponds to the DARTS edge between the second state and first state, i.e., takes the output of first op at index 1
    (0, ), # 4th op corresponds to the DARTS edge between the third state and input state
    (1, ), # 5th op corresponds to the DARTS edge between the third state and first state
    (2, 3), # 6th op takes in the 2nd and 3rd op's outputs
)
