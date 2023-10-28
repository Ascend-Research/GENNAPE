import sys
import json
from params import *
from utils.misc_utils import UniqueList
from model_src.search_space.inception.arch_sampler import sample_net_config
from model_src.search_space.inception.network import InceptionNet as TorchNet
from model_src.search_space.inception.network_tf import InceptionNet as TFNet


# Temp path swap for resolving name conflict on constant.py
# sys.path should not contain too many items, or do not import this file repeatedly
from params import BASE_DIR
__base_dir_idx = 0
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
else:
    for __di, __dn in enumerate(sys.path):
        if __dn == BASE_DIR:
            __base_dir_idx = __di
            break
    if __base_dir_idx != 0:
        sys.path[0], sys.path[__base_dir_idx] = sys.path[__base_dir_idx], sys.path[0]
from model_src.comp_graph.tf_comp_graph import ComputeGraph, OP2I
from model_src.predictor.gpi_family_data_manager import get_domain_configs
if __base_dir_idx != 0:
    sys.path[0], sys.path[__base_dir_idx] = sys.path[__base_dir_idx], sys.path[0]


def inflate_from_state_dict(sd):
    cfg = sample_net_config()
    return cfg.load_state_dict(sd)


def compare_tf_torch_networks():
    import torch
    import tensorflow.compat.v1 as tf

    def _get_nb_params_shape(shape):
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params

    def _count_number_trainable_params():
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = _get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params
    random_configs = []
    for i in range(10):
        random_configs.append(sample_net_config())

    for configs in random_configs:
        print("Sampled configs: {}".format(configs))

        # Torch
        torch_input = torch.ones((1, 3, 32, 32))
        torch_net = TorchNet(configs)
        torch_net.eval()
        with torch.no_grad():
            torch_out = torch_net(torch_input)
        n_weights = sum(p.numel() for p in torch_net.parameters() if p.requires_grad)
        print("Num trainable weights in torch model: {}".format(n_weights))

        # TF
        g = tf.Graph()
        with g.as_default():
            image_batch = tf.ones([1, 32, 32, 3], tf.float32)
            x = tf.identity(image_batch, "input")
            model = TFNet(configs)
            output = model.call(x, training=False)
            output = tf.identity(output, "output")
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25),
                                                    log_device_placement=False))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print("Num trainable params in tf model: {}".format(_count_number_trainable_params()))
        tf.reset_default_graph()
        print("")


def sample_cand_set(count=10000, log_f=print, max_n_attempts=10000,
                    output_file=P_SEP.join([CACHE_DIR, "inception_cand_data.json"])):
    net_configs = UniqueList()
    n_attempts = 0
    while len(net_configs) < count and n_attempts < max_n_attempts:
        n_attempts += 1
        if net_configs.append(sample_net_config()):
            n_attempts = 0
    log_f("{} unique nets collected".format(len(net_configs)))
    data = [c.state_dict() for c in net_configs.tolist()]
    log_f("Writing data to {}".format(output_file))
    with open(output_file, "w") as f:
        json.dump(data, f)


def build_cg(sd, net_idx, H=32, W=32):
    from functools import partial
    from model_src.comp_graph.tf_comp_graph import ComputeGraph, OP2I
    from model_src.search_space.inception.network_tf import InceptionNet
    from model_src.predictor.gpi_family_data_manager import get_domain_configs

    net_config = inflate_from_state_dict(sd)
    domain_configs = get_domain_configs()
    op2idx = OP2I().build_from_file()

    def _model_maker(_configs):
        _model = InceptionNet(_configs, name="InceptionNet{}".format(net_idx))
        return lambda _x, training: _model.call(_x, training=training)

    cg = ComputeGraph(name="InceptionNet{}".format(net_idx),
                      H=H, W=W,
                      max_kernel_size=domain_configs["max_kernel_size"],
                      max_hidden_size=domain_configs["max_hidden_size"],
                      max_derived_H=domain_configs["max_h"],
                      max_derived_W=domain_configs["max_w"])
    cg.build_from_model_maker(partial(_model_maker, _configs=net_config),
                              op2idx, oov_threshold=0.)
    # cg.gviz_visualize()
    return cg


def gen_labelled_cg_data(labelled_json, output_file, cg_maker,
                         H=32, W=32):
    from tqdm import tqdm
    from model_src.comp_graph.tf_comp_graph import get_cg_state_dict
    with open(labelled_json, "r") as f:
        data = json.load(f)
    net_idx = 0
    for k, v in tqdm(data.items(), desc="Building cg data", ascii=True):
        cg = cg_maker(v["net_config"], net_idx, H, W)
        v["cg"] = get_cg_state_dict(cg)
        net_idx += 1
    print("Writing {} labelled cg data to {}".format(len(data), output_file))
    with open(output_file, "w") as f:
        json.dump(data, f)


def visualize_labelled_cg_data(data_file):
    import random
    from scipy.stats import spearmanr
    from model_src.comp_graph.tf_comp_graph import load_from_state_dict

    with open(data_file, "r") as f:
        data = json.load(f)
    print("Loaded {} instances".format(len(data)))

    data = sorted([v for k, v in data.items()], key=lambda d:d["max_perf"], reverse=True)

    best_arch = data[0]
    worst_arch = data[-1]

    best_cg = ComputeGraph(name="Best", H=32, W=32, C_in=3,
                           max_hidden_size=512, max_kernel_size=7,
                           max_derived_H=256, max_derived_W=256)
    best_cg = load_from_state_dict(best_cg, best_arch["cg"])
    # best_cg.gviz_visualize(filename="Best")
    best_feat = best_cg.get_gnn_features()
    print(best_feat)

    worst_cg = ComputeGraph(name="Worst", H=32, W=32, C_in=3,
                            max_hidden_size=512, max_kernel_size=7,
                            max_derived_H=256, max_derived_W=256)
    worst_cg = load_from_state_dict(worst_cg, worst_arch["cg"])
    # worst_cg.gviz_visualize(filename="Worst")
    worst_feat = worst_cg.get_gnn_features()
    print(worst_feat)

    rand_arch = random.choice(data)
    rand_cg = ComputeGraph(name="Random", H=32, W=32, C_in=3,
                           max_hidden_size=512, max_kernel_size=7,
                           max_derived_H=256, max_derived_W=256)
    rand_cg = load_from_state_dict(rand_cg, rand_arch["cg"])
    # rand_cg.gviz_visualize(filename="Random")
    rand_feat = rand_cg.get_gnn_features()
    print(rand_feat)

    truth_perfs = [d["max_perf"] for d in data]
    pt_perfs = [d["perfs"][-1] for d in data]
    spr, spp = spearmanr(pt_perfs, truth_perfs)
    print("PT-max spearman rho: {}".format(spr))

    truth_perfs = [d["perfs"][-1] for d in data]
    pt_perfs = [d["perfs"][-1] for d in data]
    spr, spp = spearmanr(pt_perfs, truth_perfs)
    print("PT-last spearman rho: {}".format(spr))


if __name__ == "__main__":

    # compare_tf_torch_networks()

    # _net1 = sample_net_config()
    # _net2 = inflate_from_state_dict(_net1.state_dict())
    # assert str(_net2) == str(_net1)

    # sample_cand_set()

    # gen_labelled_cg_data(P_SEP.join([DATA_DIR, "inception_cifar10_eval_results.json"]),
    #                      P_SEP.join([DATA_DIR, "inception_cifar10_labelled_cg_data.json"]),
    #                      build_cg, H=32, W=32)
    # visualize_labelled_cg_data(P_SEP.join([DATA_DIR, "inception_cifar10_labelled_cg_data.json"]))

    print("done")
