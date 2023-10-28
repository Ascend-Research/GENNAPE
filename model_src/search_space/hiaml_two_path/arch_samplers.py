import random
from utils.misc_utils import sample_categorical_idx
from model_src.search_space.hiaml_two_path.constants import *
from model_src.search_space.hiaml_two_path.networks import TwoPathNet
from model_src.search_space.hiaml_two_path.arch_configs import TwoPathNetConfig, SinglePathBlockConfig


"""
For two path nets:
Every block must contain at least 1 conv op
Every block can contain at most 1 supp op
"""


def sample_block_ops(n_ops, conv_ops, supp_ops):
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
    return ops, op_types


def sample_sp_block_config(max_n_ops, min_n_ops,
                           conv_ops, supp_ops):
    n_ops = random.choice(list(range(min_n_ops, max_n_ops + 1)))
    ops, op_types = sample_block_ops(n_ops, conv_ops, supp_ops)
    config = SinglePathBlockConfig(ops, op_types)
    assert min_n_ops <= len(config) <= max_n_ops
    return config


def sample_two_path_net_config(max_path_len=TWO_PATH_MAX_N_BLOCKS_PER_PATH,
                               min_path_len=TWO_PATH_MIN_N_BLOCKS_PER_PATH,
                               max_n_ops=TWO_PATH_MAX_N_OPS_PER_BLOCK,
                               min_n_ops=TWO_PATH_MIN_N_OPS_PER_BLOCK,
                               num_reduction_per_path=TWO_PATH_NUM_REDUCTIONS_PER_PATH,
                               conv_ops=TWO_PATH_CONV_OPS, supp_ops=TWO_PATH_SUPP_OPS):
    path1_len = random.choice(list(range(min_path_len, max_path_len + 1)))
    path1 = [sample_sp_block_config(max_n_ops, min_n_ops,
                                    conv_ops, supp_ops)
             for _ in range(path1_len)]
    path1_reduce_inds = list(range(len(path1)))
    random.shuffle(path1_reduce_inds)
    assert len(path1_reduce_inds) >= num_reduction_per_path
    path1_reduce_inds = path1_reduce_inds[:num_reduction_per_path]
    path1_reduce_inds.sort()

    path2_len = random.choice(list(range(min_path_len, max_path_len + 1)))
    path2 = [sample_sp_block_config(max_n_ops, min_n_ops,
                                    conv_ops, supp_ops)
             for _ in range(path2_len)]
    path2_reduce_inds = list(range(len(path2)))
    random.shuffle(path2_reduce_inds)
    assert len(path2_reduce_inds) >= num_reduction_per_path
    path2_reduce_inds = path2_reduce_inds[:num_reduction_per_path]
    path2_reduce_inds.sort()

    config = TwoPathNetConfig(path1, path2, path1_reduce_inds, path2_reduce_inds)
    return config


def test_two_path_net_sampler(n=100000):
    import torch
    from tqdm import tqdm
    from thop import profile
    from utils.misc_utils import RunningStatMeter

    unique_nets = set()
    bar = tqdm(total=n, desc="Testing sampler", ascii=True)
    flops_meter = RunningStatMeter()
    n_params_meter = RunningStatMeter()
    batch = torch.ones(1, 3, 32, 32)
    while len(unique_nets) < n:
        net_config = sample_two_path_net_config()
        if str(net_config) in unique_nets:
            continue
        unique_nets.add(str(net_config))
        model = TwoPathNet(net_config)
        _macs, _n_params = profile(model, inputs=(batch, ),
                                   verbose=False)
        flops_meter.update(_macs * 2 / 1e9)
        n_params_meter.update(_n_params / 1e6)
        bar.desc = "Testing sampler, avg flops: {:.5}G, n_params: {:.5}M".format(flops_meter.avg,
                                                                                 n_params_meter.avg)
        bar.update(1)
    bar.close()


def sample_hiaml_net_config(n_stages=4, block_cands=HIAML_OPS):
    return [random.choice(block_cands) for _ in range(n_stages)]


if __name__ == "__main__":

    # test_two_path_net_sampler()

    print("done")
