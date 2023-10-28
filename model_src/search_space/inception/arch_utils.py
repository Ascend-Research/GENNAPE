from model_src.search_space.inception.constants import SUPP_C_CHANGE_CAPABLE_OPS


def make_divisible(v, divisor=16, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by divisor
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_boundary_conv_op_idx(op_types, reverse=False):
    if reverse:
        for ni, t in enumerate(op_types[::-1]):
            if t == "conv":
                return len(op_types) - 1 - ni
    else:
        for ni, t in enumerate(op_types):
            if t == "conv":
                return ni
    assert False, "No conv op found in: {}".format(op_types)


def get_boundary_C_change_op_idx(op_names, op_types, reverse=False):
    if reverse:
        for ni, t in enumerate(op_types[::-1]):
            if t == "conv" or op_names[::-1][ni] in SUPP_C_CHANGE_CAPABLE_OPS:
                return len(op_types) - 1 - ni
    else:
        for ni, t in enumerate(op_types):
            if t == "conv" or op_names[ni] in SUPP_C_CHANGE_CAPABLE_OPS:
                return ni
    assert False, "No c change capable op found in: {}".format(op_types)


def can_have_C_change(ops, op_types):
    idx1 = get_boundary_C_change_op_idx(ops, op_types)
    idx2 = get_boundary_C_change_op_idx(ops, op_types, reverse=True)
    return idx1 != idx2
