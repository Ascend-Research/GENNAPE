import os
import json
import copy
import pickle
import random
from tqdm import tqdm
from functools import partial
from utils.misc_utils import RunningStatMeter
from params import P_SEP, CACHE_DIR, DATA_DIR
from model_src.comp_graph.tf_comp_graph import ComputeGraph, OP2I, load_from_state_dict


_DOMAIN_CONFIGS = {
    "classification": { # NOTE: these settings would apply to all families
        "c_in": 3,
        "max_h": 256,
        "max_w": 256,
        "max_kernel_size": 7,
        "max_hidden_size": 512, # May not need to be too large to give normalized values explicitly between 0 and 1
    },
}


_FAMILY2MAX_HW = {
    "ofa": 224,
    "nb101": 32,
    "nb301": 32,
    "nb201c10": 32,
    "nb201c100": 32,
    "nb201imgnet": 32,
}


def get_domain_configs(domain="classification"):
    return _DOMAIN_CONFIGS[domain]


def reset_cg_max_HW(input_file, output_file,
                    max_H, max_W,
                    log_f=print):
    log_f("Loading cache from: {}".format(input_file))
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    new_data = []
    for di, d in enumerate(data):
        cg = d["compute graph"]
        assert isinstance(cg, ComputeGraph)
        if di == 0:
            log_f("Resetting max H from {} to {}".format(cg.max_derived_H, max_H))
            log_f("Resetting max W from {} to {}".format(cg.max_derived_W, max_W))
        cg.max_derived_H = max_H
        cg.max_derived_W = max_W
        new_data.append(d)
    log_f("Writing {} compute graph data to {}".format(len(new_data), output_file))
    with open(output_file, "wb") as f:
        pickle.dump(new_data, f, protocol=4)


def _build_cache_ofa_pn_mbv3(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_ofa_pn_mbv3_src_data
    from model_src.search_space.ofa_profile.networks_tf import OFAMbv3Net, OFAProxylessNet

    # Make sure duplicate checking is done before this!
    op2idx = OP2I().build_from_file()

    def _model_maker(_configs, _net_func, _name):
        _model = _net_func(_configs, name=_name)
        return lambda _x, training: _model.call(_x, training=training)

    def _single_builder(_configs, _name, _net_func, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(_model_maker, _configs=_configs,
                                           _net_func=_net_func, _name=_name),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_ofa_pn_mbv3_src_data()
    # TODO: consider a chunk write in the case of large data size
    cache_data = []
    bar = tqdm(total=len(data), desc="Building OFA-PN/MBV3 comp graph cache", ascii=True)
    for ni, ((res, net_config), acc, flops, n_params) in enumerate(data):
        net_config_list = copy.deepcopy(net_config)
        if net_config[0][0].startswith("mbconv2"):
            cg = _single_builder(net_config, "OFA-PN-Net{}".format(ni), OFAProxylessNet,
                                 res, res)
        elif net_config[0][0].startswith("mbconv3"):
            cg = _single_builder(net_config, "OFA-MBV3-Net{}".format(ni), OFAMbv3Net,
                                 res, res)
        else:
            raise ValueError("Invalid net configs of OFA: {}".format(net_config))
        cache_data.append({
            "compute graph": cg,
            "acc": acc / 100.,
            "flops": flops,
            "n_params": n_params,
            "original config": (res, net_config_list),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} OFA-PN/MBV3 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_nb301(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_nb301_src_data
    from model_src.darts.model_darts_tf import cifar10_model_maker

    # Make sure duplicate checking is done before this!
    op2idx = OP2I().build_from_file()

    def _single_builder(_geno, _name, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(cifar10_model_maker, genotype=_geno),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_nb301_src_data()
    # TODO: consider a chunk write in the case of large data size
    cache_data = []
    bar = tqdm(total=len(data), desc="Building NB301 comp graph cache", ascii=True)
    for ni, (geno, acc, flops, n_params) in enumerate(data):
        cg = _single_builder(geno, "NB301-Net{}".format(ni), 32, 32)
        cache_data.append({
            "compute graph": cg,
            "acc": acc / 100.,
            "flops": flops,
            "n_params": n_params,
            "original config": geno,
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} NB301 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_nb101(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_nb101_src_data
    from model_src.search_space.nb101.example_nb101 import nb101_model_maker

    # Make sure duplicate checking is done before this!
    op2idx = OP2I().build_from_file()

    def _single_builder(_ops, _adj_mat, _name, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(nb101_model_maker, ops=_ops, adj_mat=_adj_mat),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_nb101_src_data()
    # TODO: consider a chunk write in the case of large data size
    cache_data = []
    bar = tqdm(total=len(data), desc="Building NB101 comp graph cache", ascii=True)
    for ni, ((ops, adj_mat), acc, flops, n_params) in enumerate(data):
        cg = _single_builder(ops, adj_mat, "NB101-Net{}".format(ni), 32, 32)
        cache_data.append({
            "compute graph": cg,
            "acc": acc,
            "flops": flops,
            "n_params": n_params,
            "original config": (ops, adj_mat),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} NB101 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_nb201(output_file,
                       src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c10_src_data.pkl"]),
                       n_classes=10, H=32, W=32, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_nb201_src_data
    from model_src.search_space.nb201.networks_tf import NB201Net

    log_f("Building NB201 comp graph cache from {}".format(src_file))
    log_f("Number of classes: {}".format(n_classes))
    log_f("H: {}, W: {}".format(H, W))

    op2idx = OP2I().build_from_file()

    def _model_maker(_ops, _input_inds):
        _net = NB201Net(_ops, _input_inds,
                        n_classes=n_classes)
        return _net

    def _single_builder(_ops, _input_inds, _name, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(_model_maker, _ops=_ops, _input_inds=_input_inds),
                                   op2idx, oov_threshold=0.)
        assert len(_cg.nodes) > 10, "Found potentially invalid cg: {}, ops: {}".format(str(_cg), _ops)
        return _cg

    data = load_gpi_nb201_src_data(src_file)
    cache_data = []
    bar = tqdm(total=len(data), desc="Building NB201 comp graph cache", ascii=True)
    for ni, ((ops, op_input_inds), acc, flops) in enumerate(data):
        cg = _single_builder(ops, op_input_inds, "NB201-Net{}".format(ni), H, W)
        cache_data.append({
            "compute graph": cg,
            "acc": acc,
            "flops": flops,
            "original config": (ops, op_input_inds),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} NB201 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_ofa_resnet(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_ofa_resnet_src_data
    from model_src.search_space.ofa_profile.networks_tf import OFAResNet

    op2idx = OP2I().build_from_file()

    def _model_maker(_configs, _net_func, _name):
        _model = _net_func(_configs, name=_name)
        return lambda _x, training: _model.call(_x, training=training)

    def _single_builder(_configs, _name, _net_func, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(_model_maker, _configs=_configs,
                                           _net_func=_net_func, _name=_name),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_ofa_resnet_src_data()
    # TODO: consider a chunk write in the case of large data size
    cache_data = []
    bar = tqdm(total=len(data), desc="Building OFA-ResNet comp graph cache", ascii=True)
    for ni, ((res, net_config), acc, flops, n_params) in enumerate(data):
        cg = _single_builder(copy.deepcopy(net_config),
                             "OFA-ResNet{}".format(ni), OFAResNet,
                             res, res)
        cache_data.append({
            "compute graph": cg,
            "acc": acc / 100.,
            "flops": flops,
            "n_params": n_params,
            "original config": (res, net_config),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} OFA-ResNet compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


class FamilyDataManager:
    """
    Family-based data manager for the Generalizable Predictor Interface
    Prepares train/dev/test data for each family and combines them
    Also responsible for caching compute graphs
    """
    def __init__(self, families=("nb101", "nb301", "ofa"),
                 family2args=None,
                 cache_dir=CACHE_DIR, data_dir=DATA_DIR,
                 log_f=print):
        self.log_f = log_f
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.families = families
        self.family2args = family2args
        self.validate_cache()

    def get_cache_file_path(self, family):
        return P_SEP.join([self.cache_dir, "gpi_{}_comp_graph_cache.pkl".format(family)])

    def _build_cache(self, family, cache_file):
        if family.lower() == "ofa": # TODO: re-factor to remove ifs
            _build_cache_ofa_pn_mbv3(cache_file, self.log_f)
        elif family.lower() == "ofa_resnet":
            _build_cache_ofa_resnet(cache_file, self.log_f)
        elif family.lower() == "nb301":
            _build_cache_nb301(cache_file, self.log_f)
        elif family.lower() == "nb101":
            _build_cache_nb101(cache_file, self.log_f)
        elif family.lower() == "nb201c10": # Only 4096 instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c10_src_data.pkl"]),
                               n_classes=10, H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201c100": # Only 4096 instances
            _build_cache_nb201(cache_file, 
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c100_src_data.pkl"]),
                               n_classes=100,H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201imgnet": # Only 4096 instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201imgnet_src_data.pkl"]),
                               n_classes=120, H=16, W=16,
                               log_f=self.log_f)
        elif family.lower() == "nb201c10_complete": # Full 15k instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c10_complete_src_data.pkl"]),
                               n_classes=10, H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201c100_complete": # Full 15k instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c100_complete_src_data.pkl"]),
                               n_classes=100, H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201imgnet_complete": # Full 15k instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201imgnet_complete_src_data.pkl"]),
                               n_classes=120, H=16, W=16,
                               log_f=self.log_f)
        else:
            raise ValueError("Unknown family: {}".format(family))

    def validate_cache(self):
        # If compute graph cache is not available for some families, build it
        for f in self.families:

            if f.lower() == "hiaml" or f.lower() == "two_path" or \
                    f.lower() == "inception":
                continue # These families loads from json files

            if f.lower() == "ofa_mbv3" or f.lower() == "ofa_pn":
                f = "ofa" # These are subspaces

            cache_file = self.get_cache_file_path(f)
            if not os.path.isfile(cache_file):
                self.log_f("Building cache for {}".format(f))
                self._build_cache(f, cache_file)

        self.log_f("Cache validated for {}".format(self.families))

    def load_cache_data(self, family):
        if family.lower() == "hiaml" or family.lower() == "two_path":
            # These two families loads from json files
            d = self.get_gpi_custom_set(family_name=family.lower(),
                                        perf_diff_threshold=0,
                                        target_round_n=None,
                                        verbose=False)
            data = [{"compute graph": t[0], "acc": t[1]} for t in d]
        elif family.lower() == "inception":
            d = self.get_inception_custom_set(perf_diff_threshold=0,
                                              target_round_n=None,
                                              verbose=False)
            data = [{"compute graph": t[0], "acc": t[1]} for t in d]
        elif family.lower() == "ofa_mbv3":
            cache_file = self.get_cache_file_path("ofa")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            subset = []
            for d in data:
                if "mbv3" in d["compute graph"].name.lower():
                    subset.append(d)
            assert len(subset) > 0, "Found empty subset for ofa_mbv3"
            return subset
        elif family.lower() == "ofa_pn":
            cache_file = self.get_cache_file_path("ofa")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            subset = []
            for d in data:
                if "pn" in d["compute graph"].name.lower():
                    subset.append(d)
            assert len(subset) > 0, "Found empty subset for ofa_pn"
            return subset
        else:
            cache_file = self.get_cache_file_path(family)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
        return data

    @staticmethod
    def override_cg_max_attrs(data, max_H=None, max_W=None,
                              max_hidden=None, max_kernel=None):
        """
        In-place override of some common cg global attributes
        NOTE: ensure the attribute is not used in any pre-computed features
        """
        for d in data:
            cg = d["compute graph"]
            assert isinstance(cg, ComputeGraph)
            if max_H is not None:
                cg.max_derived_H = max_H
            if max_W is not None:
                cg.max_derived_W = max_W
            if max_hidden is not None:
                cg.max_hidden_size = max_hidden
            if max_kernel is not None:
                cg.max_kernel_size = max_kernel

    def get_src_train_dev_test_sets(self, dev_ratio, test_ratio,
                                    group_by_family=False, shuffle=False,
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    verbose=True):
        # Returns the combined data dicts
        family2data = {}
        for f in self.families:
            if verbose:
                self.log_f("Loading {} cache data...".format(f))
            fd = self.load_cache_data(f)

            if verbose:
                self.log_f("Specified normalize_HW_per_family={}".format(normalize_HW_per_family))
            if normalize_HW_per_family:
                self.override_cg_max_attrs(fd,
                                           max_H=_FAMILY2MAX_HW[f],
                                           max_W=_FAMILY2MAX_HW[f])
            if max_hidden_size is not None:
                if verbose:
                    self.log_f("Override max_hidden_size to {}".format(max_hidden_size))
                self.override_cg_max_attrs(fd, max_hidden=max_hidden_size)
            if max_kernel_size is not None:
                if verbose:
                    self.log_f("Override max_kernel_size to {}".format(max_kernel_size))
                self.override_cg_max_attrs(fd, max_kernel=max_kernel_size)

            if shuffle:
                random.shuffle(fd)

            if self.family2args is not None and \
                "max_size" in self.family2args and \
                f in self.family2args["max_size"]:
                max_size = self.family2args["max_size"][f]
                fd = fd[:max_size]
                if verbose:
                    self.log_f("Specified max total size for {}: {}".format(f, max_size))

            dev_size = max(int(dev_ratio * len(fd)), 1)
            test_size = max(int(test_ratio * len(fd)), 1)
            dev_data = fd[:dev_size]
            test_data = fd[dev_size:dev_size + test_size]
            train_data = fd[dev_size + test_size:]
            if test_ratio < 1e-5:
                # If test_ratio == 0, assume there's no test data and won't care about test performance
                # Simply merge train/test data
                train_data += test_data
                self.log_f("Test ratio: {} too small, will add test data to train data".format(test_ratio))
            family2data[f] = (train_data, dev_data, test_data)
            if verbose:
                self.log_f("Family {} train size: {}".format(f, len(train_data)))
                self.log_f("Family {} dev size: {}".format(f, len(dev_data)))
                self.log_f("Family {} test size: {}".format(f, len(test_data)))
        if group_by_family:
            return family2data
        else:
            train_set, dev_set, test_set = [], [], []
            for f, (train, dev, test) in family2data.items():
                train_set.extend(train)
                dev_set.extend(dev)
                test_set.extend(test)
            random.shuffle(train_set)
            random.shuffle(dev_set)
            random.shuffle(test_set)
            if verbose:
                self.log_f("Combined train size: {}".format(len(train_set)))
                self.log_f("Combined dev size: {}".format(len(dev_set)))
                self.log_f("Combined test size: {}".format(len(test_set)))
            return train_set, dev_set, test_set

    def get_regress_train_dev_test_sets(self, dev_ratio, test_ratio,
                                        normalize_target=False,
                                        normalize_max=None,
                                        group_by_family=False,
                                        normalize_HW_per_family=False,
                                        max_hidden_size=None, max_kernel_size=None,
                                        shuffle=False, perf_key="acc",
                                        verbose=True):
        if group_by_family:
            # Returns a dict of family_str : (train, dev, test)
            family2data = self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                           group_by_family=group_by_family,
                                                           normalize_HW_per_family=normalize_HW_per_family,
                                                           max_hidden_size=max_hidden_size,
                                                           max_kernel_size=max_kernel_size,
                                                           shuffle=shuffle, verbose=verbose)
            tgt_meter = RunningStatMeter()
            rv = {}
            for f, (train_data, dev_data, test_data) in family2data.items():
                fam_tgt_meter = RunningStatMeter()
                train_set, dev_set, test_set = [], [], []
                for d in train_data:
                    train_set.append([d["compute graph"], d[perf_key]])
                    tgt_meter.update(d[perf_key])
                    fam_tgt_meter.update(d[perf_key])
                for d in dev_data:
                    dev_set.append([d["compute graph"], d[perf_key]])
                    tgt_meter.update(d[perf_key])
                    fam_tgt_meter.update(d[perf_key])
                for d in test_data:
                    test_set.append([d["compute graph"], d[perf_key]])
                rv[f] = (train_set, dev_set, test_set)
                if verbose:
                    self.log_f("Max {} target value: {}".format(f, fam_tgt_meter.max))
                    self.log_f("Min {} target value: {}".format(f, fam_tgt_meter.min))
                    self.log_f("Avg {} target value: {}".format(f, fam_tgt_meter.avg))

            if verbose:
                self.log_f("Max global target value: {}".format(tgt_meter.max))
                self.log_f("Min global target value: {}".format(tgt_meter.min))
                self.log_f("Avg global target value: {}".format(tgt_meter.avg))

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for _, (train_set, dev_set, test_set) in rv.items():
                    for t in train_set:
                        t[-1] /= tgt_meter.max
                    for t in dev_set:
                        t[-1] /= tgt_meter.max
                    for t in test_set:
                        t[-1] /= tgt_meter.max
                        if normalize_max is not None:
                            t[-1] = min(t[-1], normalize_max)

            return rv

        else:
            # Each instance is (compute graph, perf value)
            train_dicts, dev_dicts, test_dicts = \
                self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                 group_by_family=group_by_family,
                                                 shuffle=shuffle, verbose=verbose)
            tgt_meter = RunningStatMeter()
            train_set, dev_set, test_set = [], [], []
            for d in train_dicts:
                train_set.append([d["compute graph"], d[perf_key]])
                tgt_meter.update(d[perf_key])
            for d in dev_dicts:
                dev_set.append([d["compute graph"], d[perf_key]])
                tgt_meter.update(d[perf_key])
            for d in test_dicts:
                test_set.append([d["compute graph"], d[perf_key]])

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for t in train_set:
                    t[-1] /= tgt_meter.max
                for t in dev_set:
                    t[-1] /= tgt_meter.max
                for t in test_set:
                    t[-1] /= tgt_meter.max
                    if normalize_max is not None:
                        t[-1] = min(t[-1], normalize_max)
            if verbose:
                self.log_f("Max target value: {}".format(tgt_meter.max))
                self.log_f("Min target value: {}".format(tgt_meter.min))
                self.log_f("Avg target value: {}".format(tgt_meter.avg))

            return train_set, dev_set, test_set

    def get_nb201_test_set(self, family_name="nb201c10",
                           n_nets=None, ordered=False,
                           normalize_HW_per_family=False,
                           max_hidden_size=None, max_kernel_size=None,
                           perf_diff_threshold=2e-4,
                           perf_key="acc", verbose=True):
        if verbose:
            self.log_f("Loading {} cache data...".format(family_name))

        fd = self.load_cache_data(family_name)

        if verbose:
            self.log_f("Specified normalize_HW_per_family={}".format(normalize_HW_per_family))
        if normalize_HW_per_family:
            self.override_cg_max_attrs(fd,
                                       max_H=_FAMILY2MAX_HW[family_name],
                                       max_W=_FAMILY2MAX_HW[family_name])
        if max_hidden_size is not None:
            if verbose:
                self.log_f("Override max_hidden_size to {}".format(max_hidden_size))
            self.override_cg_max_attrs(fd, max_hidden=max_hidden_size)
        if max_kernel_size is not None:
            if verbose:
                self.log_f("Override max_kernel_size to {}".format(max_kernel_size))
            self.override_cg_max_attrs(fd, max_kernel=max_kernel_size)

        if ordered:
            if verbose: self.log_f("Specified ordered={}".format(ordered))
            fd.sort(key=lambda _d:_d[perf_key], reverse=True)

        if n_nets is not None:
            if verbose: self.log_f("Specified num nets: {}".format(n_nets))
            fd = fd[:n_nets]

        rv = [[_d["compute graph"], _d[perf_key]] for _d in fd]

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[1], reverse=True)
            pruned_indices = set()
            for i, (g, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for {}: {}".format(family_name, tgt_meter.max))
            self.log_f("Min final target value for {}: {}".format(family_name, tgt_meter.min))
            self.log_f("Avg final target value for {}: {}".format(family_name, tgt_meter.avg))
            self.log_f("Loaded {} {} instances".format(len(rv), family_name))

        return rv

    def get_gpi_custom_set(self, family_name="hiaml", dataset="cifar10",
                           max_hidden_size=None, max_kernel_size=None,
                           perf_diff_threshold=2e-4, target_round_n=None,
                           verbose=True):
        if verbose:
            self.log_f("Loading {} data...".format(family_name))

        data_file = P_SEP.join([self.data_dir, "gpi_test_{}_{}_labelled_cg_data.json".format(family_name, dataset)])

        with open(data_file, "r") as f:
            data = json.load(f)

        if max_hidden_size is None:
            max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
        if max_kernel_size is None:
            max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
        max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
        max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]

        rv = []
        bar = None
        if verbose:
            bar = tqdm(total=len(data), desc="Inflating compute graphs", ascii=True)
        for k, v in data.items():
            cg = ComputeGraph(name="", H=32, W=32, C_in=3,
                              max_hidden_size=max_hidden_size, max_kernel_size=max_kernel_size,
                              max_derived_H=max_derived_H, max_derived_W=max_derived_W)
            cg = load_from_state_dict(cg, v["cg"])
            acc = v["max_perf"] / 100.
            rv.append((cg, acc))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[1], reverse=True)
            pruned_indices = set()
            for i, (g, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if target_round_n is not None:
            if verbose: self.log_f("Specified target round n={}".format(target_round_n))
            rv = [(c, round(t, target_round_n)) for c, t in rv]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for {}: {}".format(family_name, tgt_meter.max))
            self.log_f("Min final target value for {}: {}".format(family_name, tgt_meter.min))
            self.log_f("Avg final target value for {}: {}".format(family_name, tgt_meter.avg))
            self.log_f("Loaded {} {} instances".format(len(rv), family_name))

        return rv

    def get_inception_custom_set(self, dataset="cifar10",
                                 max_hidden_size=None, max_kernel_size=None,
                                 perf_diff_threshold=None, target_round_n=None,
                                 verbose=True):

        data_file = P_SEP.join([self.data_dir, "inception_{}_labelled_cg_data.json".format(dataset)])

        with open(data_file, "r") as f:
            data = json.load(f)

        if max_hidden_size is None:
            max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
        if max_kernel_size is None:
            max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
        max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
        max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]

        rv = []
        bar = None
        if verbose:
            bar = tqdm(total=len(data), desc="Inflating compute graphs", ascii=True)
        for k, v in data.items():
            cg = ComputeGraph(name="", H=32, W=32, C_in=3,
                              max_hidden_size=max_hidden_size, max_kernel_size=max_kernel_size,
                              max_derived_H=max_derived_H, max_derived_W=max_derived_W)
            cg = load_from_state_dict(cg, v["cg"])
            acc = v["max_perf"] / 100.
            rv.append((cg, acc))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[1], reverse=True)
            pruned_indices = set()
            for i, (g, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if target_round_n is not None:
            if verbose: self.log_f("Specified target round n={}".format(target_round_n))
            rv = [(c, round(t, target_round_n)) for c, t in rv]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for inception: {}".format(tgt_meter.max))
            self.log_f("Min final target value for inception: {}".format(tgt_meter.min))
            self.log_f("Avg final target value for inception: {}".format(tgt_meter.avg))
            self.log_f("Loaded {} inception instances".format(len(rv)))

        return rv
