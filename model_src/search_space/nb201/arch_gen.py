import pickle
import collections
from params import *
from tqdm import tqdm
import tensorflow as tf
from utils.misc_utils import map_to_range
from model_src.search_space.nb201.networks_tf import NB201Net
from model_src.search_space.nb201.constants import OP_INPUT_INDS


def get_cg_compatible_nb201_keys(keys):
    """
    A meaningful net is a network that does not contain any none (zero) ops
    :return: a list of meaningful net keys
    """
    rv = []
    for k in keys:
        ops = nb201_key2labels(k)
        if not any(op == "none" for op in ops):
            rv.append(k)
    return rv


def load_complete_nb201(data_file=P_SEP.join([DATA_DIR, "NB201-v1_1-096897.pth"]), log_f=print,
                        d_set="cifar10", eval_method="ori-test", show_example=False,
                        simple_data_dict_cache=P_SEP.join([DATA_DIR, "nb201_acc_flops_dict.pkl"])):
    from nas_201_api import NASBench201API as API
    api = API(data_file)
    log_f("{} archs loaded from NasBench201".format(len(api)))
    keys = set()
    max_test_acc, min_test_acc = 0., 1.
    max_flops, min_flops = 0., float("inf")
    avg_test_acc = 0.
    avg_flops = 0
    simple_data_dict = {}
    best_block_key = None
    for i in tqdm(range(len(api)), desc="Processing", ascii=True):
        info = api.query_meta_info_by_index(i)
        keys.add(info.arch_str)
        res_metrics = info.get_metrics(d_set, eval_method)
        cost_metrics = info.get_compute_costs(d_set)  # flops, params, latency
        test_acc = res_metrics["accuracy"] / 100.
        flops = cost_metrics["flops"]
        simple_data_dict[info.arch_str] = [test_acc, flops]
        if test_acc > max_test_acc:
            best_block_key = info.arch_str
        max_test_acc = max(max_test_acc, test_acc)
        min_test_acc = min(min_test_acc, test_acc)
        max_flops = max(flops, max_flops)
        min_flops = min(flops, min_flops)
        avg_test_acc += test_acc
        avg_flops += flops
    avg_test_acc /= len(api)
    avg_flops /= len(api)
    log_f("Best block key: {}".format(best_block_key))
    log_f("Max test acc: {}, min test acc: {}, avg test acc: {}".format(max_test_acc, min_test_acc, avg_test_acc))
    log_f("Max flops: {}, min flops: {}, avg flops: {}".format(max_flops, min_flops, avg_flops))
    log_f("{} unique keys recorded".format(len(keys)))
    log_f("Normalizing flops to between 0 and 1")
    for k, v in simple_data_dict.items():
        flops = v[1]
        norm_flops = map_to_range(flops, min_flops, max_flops, 0., 1.)
        v[1] = norm_flops
    for v in simple_data_dict.values():
        assert 0. <= v[0] <= 1.
        assert 0. <= v[1] <= 1.
    if simple_data_dict_cache is not None:
        with open(simple_data_dict_cache, "wb") as f:
            pickle.dump(simple_data_dict, f, protocol=4)
    if show_example:
        # Get the detailed information
        results = api.query_by_index(1, 'cifar10') # A dict of all trials for 1st net on cifar100, where the key is the seed
        print('There are {:} trials for this architecture [{:}] on cifar10'.format(len(results), api[1]))
        print("results: {}".format(str(results)))

        index = api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
        info = api.query_meta_info_by_index(index)  # This is an instance of `ArchResults`
        res_metrics = info.get_metrics('cifar10', 'ori-test')  # This is a dict with metric names as keys
        print("ori-test res_metrics: {}".format(res_metrics))

        res_metrics = info.get_metrics('cifar10', 'train')  # This is a dict with metric names as keys
        print("train res_metrics: {}".format(res_metrics))

        cost_metrics = info.get_compute_costs('cifar10')  # This is a dict with metric names as keys, e.g., flops, params, latency
        print("cost_metrics: {}".format(cost_metrics))

        # for the metric after a specific epoch
        index = api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
        api.show(index)
    return api


def get_nb201_key(node_labels, ni2de_map):
    """
    :param node_labels: Must include input/output labels
    :param ni2de_map: Mapping between node idx to darts edge pair. E.g. node idx=3: (darts_node=0, darts_node=4)
    :return: str key
    """
    darts_dst2ni = collections.defaultdict(list)
    for ni, (src, dst) in ni2de_map.items():
        darts_dst2ni[dst].append( (ni, src) )
        darts_dst2ni[dst].sort(key=lambda t: t[0])
    darts_dst_ni_pairs = [(k, v) for k, v in darts_dst2ni.items()]
    darts_dst_ni_pairs.sort(key=lambda t: t[0])
    key = ""
    for dart_dst, adj_list in darts_dst_ni_pairs:
        labels = [node_labels[ni]+"~"+str(dart_src) for ni, dart_src in adj_list]
        partial_key = "|" + "|".join(labels) + "|"
        key += partial_key + "+"
    return key[:-1]


def nb201_key2labels(key):
    """
    Reverse process of get_nb201_key()
    NOTE: does not include input and output nodes
    """
    rv = []
    node_strs = key.split("+")
    for node_str in node_strs:
        op_keys = node_str.split("|")
        op_keys = [v for v in op_keys if len(v) > 0]
        curr_src_idx = -1
        for op_key in op_keys:
            op, src_idx = op_key.split("~")
            assert int(src_idx) > curr_src_idx
            curr_src_idx = int(src_idx) # Strictly increasing
            rv.append(op)
    return rv


def test_nb201_tf_nets():
    with open(P_SEP.join([CACHE_DIR, "nb201_acc_flops_dict_c10.pkl"]), "rb") as f:
        data = pickle.load(f)
    op_input_inds = OP_INPUT_INDS

    def _model_maker(_ops, _input_inds):
        _net = NB201Net(_ops, _input_inds)
        return _net

    keys = get_cg_compatible_nb201_keys(data.keys())
    print("Found {} meaningful keys".format(len(keys)))

    for k in keys:
        ops = nb201_key2labels(k)
        assert len(op_input_inds) == len(ops)
        print("Net config: {}".format(ops))
        run_meta = tf.RunMetadata()
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25),
                                                    log_device_placement=False))
            image_batch = tf.ones([1, 32, 32, 3], tf.float32)
            x = tf.identity(image_batch, "input")
            model = _model_maker(ops, op_input_inds)
            output = model(x, training=False)
            output = tf.identity(output, "output")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
            flops = flops.total_float_ops if flops is not None else 0
            print("Measured flops: {}".format(flops))
            print("Truth acc: {}".format(data[k][0]))
            print("Truth flops: {}".format(data[k][1]))


if __name__ == "__main__":

    # test_nb201_tf_nets()

    # print("Loading cifar10")
    # load_complete_nb201(d_set="cifar10", eval_method="ori-test",
    #                     simple_data_dict_cache=P_SEP.join([CACHE_DIR, "nb201_acc_flops_dict_c10.pkl"])
    #                     )
    # print("Loading cifar100")
    # load_complete_nb201(d_set="cifar100", eval_method="x-test",
    #                     simple_data_dict_cache=P_SEP.join([CACHE_DIR, "nb201_acc_flops_dict_c100.pkl"])
    #                     )
    # print("Loading ImageNet16-120")
    # load_complete_nb201(d_set="ImageNet16-120", eval_method="x-test",
    #                     #simple_data_dict_cache=P_SEP.join([CACHE_DIR, "nb201_acc_flops_dict_img_net.pkl"])
    #                     )

    # _map = {1: (0, 1), 2: (0, 2), 3: (1, 2), 4: (0, 3), 5: (1, 3), 6: (2, 3)}
    # _k = get_nb201_key(["input", "nor_conv_3x3", "none", "nor_conv_1x1", "none", "nor_conv_1x1", "avg_pool_3x3", "output"],
    #                    _map)
    # print(_k)
    # _labels = nb201_key2labels(_k)
    # print(_labels)

    print("done")
