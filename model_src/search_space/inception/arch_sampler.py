import random
from utils.misc_utils import sample_categorical_idx
from model_src.search_space.inception.constants import *
from model_src.search_space.inception.arch_utils import can_have_C_change
from model_src.search_space.inception.arch_config import BlockConfig, NetConfig


"""
Every path in a block must contain at least 1 conv op
Every path in a block can contain at most 1 supp op
The total number of ops in a block must be within the max
The number of ops in a path must be higher than some threshold
"""


def sample_path(n_ops, conv_ops, supp_ops,
                C_ext, C_change_probs):
    op_types = ["conv"]
    ops = [random.choice(conv_ops)]
    conv_supp_weights = [len(conv_ops) / float(len(conv_ops) + len(supp_ops)),
                         len(supp_ops) / float(len(conv_ops) + len(supp_ops)),]
    if n_ops > 1:
        supp_added = False
        for _ in range(n_ops - 1):
            idx = sample_categorical_idx(conv_supp_weights)
            if not supp_added and idx == 1:
                op = random.choice(supp_ops)
                supp_added = True
                op_type = "supp"
            else:
                op = random.choice(conv_ops)
                op_type = "conv"
            ops.append(op)
            op_types.append(op_type)
    op_w_type = list(zip(ops, op_types))
    random.shuffle(op_w_type)
    ops = [t[0] for t in op_w_type]
    op_types = [t[1] for t in op_w_type]

    if can_have_C_change(ops, op_types):
        # Potential to apply additional internal C changes
        changes, probs = [t[0] for t in C_change_probs], \
                         [t[1] for t in C_change_probs]
        idx = sample_categorical_idx(probs)
        change = changes[idx]
        if change == "reduce2":
            C_internal = C_ext // 2
        elif change == "reduce4":
            C_internal = C_ext // 4
        elif change == "expand2":
            C_internal = C_ext * 2
        else:
            C_internal = C_ext
        C_internal = max(C_internal, MIN_CHANNEL_SIZE)
    else:
        C_internal = max(C_ext, MIN_CHANNEL_SIZE)

    return ops, op_types, C_internal


def sample_block_config(n_path_to_max_n_b_ops,
                        n_path_to_min_n_ops_per_path,
                        n_path_cands,
                        C_in, path_C_change_probs,
                        conv_ops, supp_ops):
    # First sample number of paths
    n_paths = random.choice(n_path_cands)
    assert n_paths <= 4

    max_n_ops_per_block = n_path_to_max_n_b_ops[n_paths]
    min_n_ops_per_path = n_path_to_min_n_ops_per_path[n_paths]

    # Determine the I/O channel sizes per path
    if n_paths == 3:
        C = C_in // 4
        C_remain = C_in - C * 4
        path_C_vals = [2 * C, C, C + C_remain]
    else:
        C = C_in // n_paths
        C_remain = C_in - C * n_paths
        path_C_vals = [C for _ in range(n_paths)]
        path_C_vals[-1] += C_remain
    random.shuffle(path_C_vals)

    # Randomly determine the distribution of ops among paths
    path_ops, path_op_types = [], []
    sel_path_channels = []
    for pi in range(n_paths):
        # First minus existing op counts
        max_n_op_per_path = max_n_ops_per_block - sum(len(v) for v in path_ops)
        # Also save enough for the remaining paths
        max_n_op_per_path -= min_n_ops_per_path * (n_paths - pi - 1)

        n_op_cands_per_path = list(range(min_n_ops_per_path,
                                         max_n_op_per_path + 1))
        n_ops = random.choice(n_op_cands_per_path)
        C_io = max(path_C_vals[pi], MIN_CHANNEL_SIZE)
        ops, op_types, C_internal = \
            sample_path(n_ops=n_ops,
                        conv_ops=conv_ops, supp_ops=supp_ops, C_ext=C_io,
                        C_change_probs=path_C_change_probs)
        path_ops.append(ops)
        path_op_types.append(op_types)
        sel_path_channels.append((C_internal, C_io))
        assert 1 <= len(ops) <= max_n_op_per_path
        if pi != n_paths - 1:
            assert max_n_op_per_path >= 1, "Ran out of ops budget"

    # Build path configs
    paths = []
    for pi in range(len(path_ops)):
        path_data = {
            "ops": path_ops[pi],
            "op_types": path_op_types[pi],
            "C_internal": sel_path_channels[pi][0],
            "C_out": sel_path_channels[pi][1],
        }
        paths.append(path_data)

    config = BlockConfig(paths)
    assert 1 <= len(config) <= max_n_ops_per_block
    return config


def sample_net_config(n_stages=N_STAGES,
                      reduction_stage_inds=REDUCTION_STAGE_INDS,
                      C_init=C_INIT,
                      min_blocks_per_stage=MIN_BLOCKS_PER_STAGE,
                      max_blocks_per_stage=MAX_BLOCKS_PER_STAGE,
                      n_path_to_max_n_b_ops=N_PATHS_TO_MAX_BLOCK_OPS,
                      n_path_to_min_n_ops_per_path=N_PATHS_TO_MIN_N_OPS_PER_PATH,
                      n_path_cands=NUM_PATHS_IN_BLOCK,
                      path_C_change_probs=PATH_C_CHANGE_PROBS,
                      conv_ops=CONV_OPS, supp_ops=SUPP_OPS):
    stages = []
    C_curr = C_init
    for si in range(n_stages):
        is_reduce = si in reduction_stage_inds

        n_blocks = random.choice(list(range(min_blocks_per_stage,
                                            max_blocks_per_stage + 1)))

        if is_reduce:
            C_out = C_curr * 2
            stride = 2
        else:
            C_out = C_curr
            stride = 1
        block_config = sample_block_config(n_path_to_max_n_b_ops,
                                           n_path_to_min_n_ops_per_path,
                                           n_path_cands, C_out,
                                           path_C_change_probs,
                                           conv_ops, supp_ops)
        stage_data = {
            "block": block_config,
            "n_blocks": n_blocks,
            "C_in": C_curr,
            "C_out": C_out,
            "stride": stride,
        }
        stages.append(stage_data)
        C_curr = C_out
    return NetConfig(stages)


def test_sampler(n=100000):
    import torch
    from tqdm import tqdm
    from thop import profile
    from utils.misc_utils import RunningStatMeter
    from search_space.inception.network import InceptionNet
    unique_nets = set()
    bar = tqdm(total=n, desc="Testing sampler", ascii=True)
    flops_meter = RunningStatMeter()
    n_params_meter = RunningStatMeter()
    batch = torch.ones(1, 3, 32, 32)
    while len(unique_nets) < n:
        net_config = sample_net_config()
        if str(net_config) in unique_nets:
            continue
        unique_nets.add(str(net_config))
        model = InceptionNet(net_config)
        _macs, _n_params = profile(model, inputs=(batch, ),
                                   verbose=False)
        flops_meter.update(_macs * 2 / 1e9)
        n_params_meter.update(_n_params / 1e6)
        bar.desc = "Testing sampler, avg flops: {:.5}G, n_params: {:.5}M".format(flops_meter.avg,
                                                                                 n_params_meter.avg)
        bar.update(1)
    bar.close()


if __name__ == "__main__":

    test_sampler()

    print("done")

