import sys
import json
import torch
from params import *
from functools import partial
import tensorflow.compat.v1 as tf
from utils.misc_utils import UniqueList
from model_src.search_space.hiaml_two_path.arch_samplers import sample_two_path_net_config, sample_hiaml_net_config


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


def load_config_from_state_dict(sd, family):
    family = family.lower()
    if family == "two_path":
        net_config = sample_two_path_net_config()
        net_config.load_state_dict(sd)
    elif family == "hiaml":
        net_config = sd
    else:
        raise ValueError("Unknown family: {}".format(family))
    return net_config


def get_state_dict_from_config(net_config):
    return net_config.state_dict() if hasattr(net_config, "state_dict") else net_config


def sample_combined_cand_set(fam_counts=(("two_path", 10000), ("hiaml", 10000)),
                             log_f=print, max_n_attempts=10000,
                             output_file=P_SEP.join([CACHE_DIR, "gpi_test_cand_data.json"])):
    data = {}
    for fam, count in fam_counts:
        fam = fam.lower()
        net_configs = UniqueList()
        n_attempts = 0
        while len(net_configs) < count and n_attempts < max_n_attempts:
            n_attempts += 1
            if fam == "two_path":
                if net_configs.append(sample_two_path_net_config()):
                    n_attempts = 0
            elif fam == "hiaml":
                if net_configs.append(sample_hiaml_net_config()):
                    n_attempts = 0
            else:
                raise ValueError("Unknown family: {}".format(fam))
        log_f("{} unique nets collected for {} family".format(len(net_configs), fam))
        data[fam] = [get_state_dict_from_config(c) for c in net_configs.tolist()]
    log_f("Writing data to {}".format(output_file))
    with open(output_file, "w") as f:
        json.dump(data, f)


def compare_tf_torch_two_path_nets():
    from model_src.search_space.hiaml_two_path.networks import TwoPathNet as TorchNet
    from model_src.search_space.hiaml_two_path.networks_tf import TwoPathNet as TFNet

    def _get_nb_params_shape(shape):
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params

    def _count_number_trainable_params():
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape()
            current_nb_params = _get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    random_configs = []
    for i in range(100):
        random_configs.append(sample_two_path_net_config())

    for configs in random_configs:
        print("Sampled configs: {}".format(configs))

        # Torch
        torch_input = torch.ones((1, 3, 32, 32))
        torch_net = TorchNet(configs)
        torch_net.eval()
        with torch.no_grad():
            torch_out = torch_net(torch_input)
        n_torch_weights = sum(p.numel() for p in torch_net.parameters() if p.requires_grad)
        print("Num trainable weights in torch model: {}".format(n_torch_weights))

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
            n_tf_weights = _count_number_trainable_params()
            print("Num trainable params in tf model: {}".format(n_tf_weights))
        tf.reset_default_graph()
        assert n_tf_weights == n_torch_weights
        print("")


def compare_tf_torch_hiaml_nets():
    from model_src.search_space.hiaml_two_path.networks import HiAMLNet as TorchNet
    from model_src.search_space.hiaml_two_path.networks_tf import HiAMLNet as TFNet

    def _get_nb_params_shape(shape):
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params

    def _count_number_trainable_params():
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape()
            current_nb_params = _get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    random_configs = []
    for i in range(100):
        random_configs.append(sample_hiaml_net_config())

    for configs in random_configs:
        print("Sampled configs: {}".format(configs))

        # Torch
        torch_input = torch.ones((1, 3, 32, 32))
        torch_net = TorchNet(configs)
        torch_net.eval()
        with torch.no_grad():
            torch_out = torch_net(torch_input)
        n_torch_weights = sum(p.numel() for p in torch_net.parameters() if p.requires_grad)
        print("Num trainable weights in torch model: {}".format(n_torch_weights))

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
            n_tf_weights = _count_number_trainable_params()
            print("Num trainable params in tf model: {}".format(n_tf_weights))
        tf.reset_default_graph()
        assert n_tf_weights == n_torch_weights
        print("")


def build_two_path_cg(sd, net_idx, H=32, W=32):
    from model_src.search_space.hiaml_two_path.networks_tf import TwoPathNet
    net_config = load_config_from_state_dict(sd, "two_path")
    domain_configs = get_domain_configs()
    op2idx = OP2I().build_from_file()

    def _model_maker(_configs):
        _model = TwoPathNet(_configs, name="TwoPathNet{}".format(net_idx))
        return lambda _x, training: _model.call(_x, training=training)

    cg = ComputeGraph(name="TwoPathNet{}".format(net_idx),
                      H=H, W=W,
                      max_kernel_size=domain_configs["max_kernel_size"],
                      max_hidden_size=domain_configs["max_hidden_size"],
                      max_derived_H=domain_configs["max_h"],
                      max_derived_W=domain_configs["max_w"])
    cg.build_from_model_maker(partial(_model_maker, _configs=net_config),
                              op2idx, oov_threshold=0.)
    # cg.gviz_visualize()
    return cg


def build_hiaml_cg(sd, net_idx, H=32, W=32):
    from model_src.search_space.hiaml_two_path.networks_tf import HiAMLNet
    net_config = load_config_from_state_dict(sd, "hiaml")
    domain_configs = get_domain_configs()
    op2idx = OP2I().build_from_file()

    def _model_maker(_configs):
        _model = HiAMLNet(_configs, name="HiAMLNet{}".format(net_idx))
        return lambda _x, training: _model.call(_x, training=training)

    cg = ComputeGraph(name="HiAMLNet{}".format(net_idx),
                      H=H, W=W,
                      max_kernel_size=domain_configs["max_kernel_size"],
                      max_hidden_size=domain_configs["max_hidden_size"],
                      max_derived_H=domain_configs["max_h"],
                      max_derived_W=domain_configs["max_w"])
    cg.build_from_model_maker(partial(_model_maker, _configs=net_config),
                              op2idx, oov_threshold=0.)
    # cg.gviz_visualize()
    return cg


def merge_json_files(files, output_file):
    data = {}
    for f in files:
        with open(f, "r") as _f:
            d = json.load(_f)
        print("Loaded {} instances from {}".format(len(d), f))
        for k, v in d.items():
            assert k not in data, "Duplicated key found: {}".format(k)
            data[k] = v
    print("Writing {} merged data to {}".format(len(data), output_file))
    with open(output_file, "w") as f:
        json.dump(data, f)


def gen_labelled_cg_data(labelled_json, output_file, cg_maker,
                         H=32, W=32):
    from tqdm import tqdm
    from model_src.comp_graph.tf_comp_graph import get_cg_state_dict
    with open(labelled_json, "r") as f:
        data = json.load(f)
    net_idx = 0
    for k, v in tqdm(data.items(), desc="Building cg data"):
        cg = cg_maker(v["net_config"], net_idx, H, W)
        v["cg"] = get_cg_state_dict(cg)
        net_idx += 1
    print("Writing {} labelled cg data to {}".format(len(data), output_file))
    with open(output_file, "w") as f:
        json.dump(data, f)


def visualize_labelled_cg_data(data_file):
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
    best_cg.gviz_visualize(filename="Best")

    worst_cg = ComputeGraph(name="Worst", H=32, W=32, C_in=3,
                            max_hidden_size=512, max_kernel_size=7,
                            max_derived_H=256, max_derived_W=256)
    worst_cg = load_from_state_dict(worst_cg, worst_arch["cg"])
    worst_cg.gviz_visualize(filename="Worst")


if __name__ == "__main__":

    # compare_tf_torch_two_path_nets()

    # compare_tf_torch_hiaml_nets()

    # sample_combined_cand_set()

    # sample_combined_cand_set(fam_counts=(("two_path", 10000),),
    #                          output_file=P_SEP.join([CACHE_DIR, "gpi_test_cand_data_large.json"]))

    # merge_json_files([P_SEP.join([DATA_DIR, "acc_dataset", "hiaml1", "gpi_test_hiaml_cifar10_eval_results.json"]),
    #                   P_SEP.join([DATA_DIR, "acc_dataset", "hiaml2", "gpi_test_hiaml_cifar10_eval_results.json"]),
    #                   ],
    #                  P_SEP.join([DATA_DIR, "gpi_test_hiaml_cifar10_eval_results.json"]))
    # gen_labelled_cg_data(P_SEP.join([DATA_DIR, "gpi_test_hiaml_cifar10_eval_results.json"]),
    #                      P_SEP.join([DATA_DIR, "gpi_test_hiaml_cifar10_labelled_cg_data.json"]),
    #                      build_hiaml_cg, H=32, W=32)
    # gen_labelled_cg_data(P_SEP.join([DATA_DIR, "gpi_test_two_path_cifar10_eval_results.json"]),
    #                      P_SEP.join([DATA_DIR, "gpi_test_two_path_cifar10_labelled_cg_data.json"]),
    #                      build_two_path_cg, H=32, W=32)

    # visualize_labelled_cg_data(P_SEP.join([DATA_DIR, "gpi_test_hiaml_cifar10_labelled_cg_data.json"]))
    # visualize_labelled_cg_data(P_SEP.join([DATA_DIR, "gpi_test_two_path_cifar10_labelled_cg_data.json"]))

    print("done")
