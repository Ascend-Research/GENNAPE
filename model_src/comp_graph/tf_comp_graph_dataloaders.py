import math
import torch
import random
import collections
from tqdm import tqdm
from constants import *
from utils.misc_utils import UniqueDict
from utils.graph_utils import hash_module, edge_list_to_edge_matrix, edge_list_to_edge_pairs


def _get_graph_id(regular_inds, regular_shapes,
                  weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                  edge_pairs,
                  pivot_idx=-1, pivot_name="none", strategy="complete"):
    edge_list = [[t[0] for t in edge_pairs], [t[1] for t in edge_pairs]]
    graph_inds = weighted_node_inds + regular_inds
    graph_shapes = weighted_node_shapes + regular_shapes
    node_features = []
    for ni in range(len(graph_inds)):
        op_idx = str(graph_inds[ni])
        op_shape = str(graph_shapes[ni])
        if ni < len(weighted_node_kernels):
            op_kernel = str(weighted_node_kernels[ni])
        else:
            op_kernel = "0"
        if ni < len(weighted_node_bias):
            op_bias = str(weighted_node_bias[ni])
        else:
            op_bias = "0"
        node_feature = "|".join([op_idx, op_shape, op_kernel, str(op_bias), str(pivot_idx), pivot_name])
        node_features.append(node_feature)
    if strategy == "complete":
        edge_mat = edge_list_to_edge_matrix(edge_list, len(node_features))
        graph_id = hash_module(edge_mat, node_features)
    elif strategy == "simple":
        graph_id = "+".join([str(node_features), str(edge_pairs)])
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))
    return graph_id


class CGRegressDataLoader:

    def __init__(self, batch_size, data, verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.curr_batch_idx = 0
        self.batches = []
        self._build_batches(data)
        self.n_batches = len(self.batches)

    def _build_batches(self, data):
        from model_src.comp_graph.tf_comp_graph import ComputeGraph
        # Data format: (ComputeGraph object, tgt_val)
        self.batches = []

        # Partition data by the number of nodes in graph
        bins = collections.defaultdict(list)
        for g, tgt_val in data:
            assert isinstance(g, ComputeGraph)
            key = "{}|{}".format(len(g.nodes), g.regular_node_start_idx)
            bins[key].append((g, tgt_val))

        n_batches = 0
        # Compute the number of batches in total
        for _, instances in bins.items():
            n_batches += math.ceil(len(instances) / self.batch_size)

        # Build actual batches
        bar = None
        if self.verbose:
            bar = tqdm(total=n_batches, desc="Building batches", ascii=True)
        for k, data_list in bins.items():
            idx = 0
            while idx < len(data_list):
                batch_list = data_list[idx:idx + self.batch_size]
                batch_regular_inds = []
                batch_regular_shapes = []
                batch_weighted_inds = []
                batch_weighted_shapes = []
                batch_weighted_kernels = []
                batch_weighted_bias = []
                batch_edge_list = []
                batch_tgt = []
                batch_last_node_idx_list = []
                batch_unique_str_id_set = set()
                for inst in batch_list:
                    g, tgt = inst # Expected data format
                    assert isinstance(g, ComputeGraph)
                    regular_node_inds, regular_node_shapes, \
                        weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias, \
                            edge_list = g.get_gnn_features()
                    if len(regular_node_inds) > 0:
                        batch_regular_inds.append(torch.LongTensor(regular_node_inds).unsqueeze(0))
                        batch_regular_shapes.append(torch.FloatTensor(regular_node_shapes).unsqueeze(0))
                    if len(weighted_node_inds) > 0:
                        batch_weighted_inds.append(torch.LongTensor(weighted_node_inds).unsqueeze(0))
                        batch_weighted_shapes.append(torch.FloatTensor(weighted_node_shapes).unsqueeze(0))
                        batch_weighted_kernels.append(torch.LongTensor(weighted_node_kernels).unsqueeze(0))
                        batch_weighted_bias.append(torch.FloatTensor(weighted_node_bias).unsqueeze(0))
                    batch_edge_list.append(torch.LongTensor(edge_list))
                    batch_tgt.append(tgt)
                    prev_len = batch_last_node_idx_list[-1] if len(batch_last_node_idx_list) > 0 else -1
                    batch_last_node_idx_list.append(prev_len + len(weighted_node_inds) + len(regular_node_inds))
                    graph_id = _get_graph_id(regular_node_inds, regular_node_shapes,
                                             weighted_node_inds, weighted_node_shapes, weighted_node_kernels,
                                             weighted_node_bias,
                                             edge_list_to_edge_pairs(edge_list), strategy="simple")
                    batch_unique_str_id_set.add(graph_id)
                batch_tgt = torch.FloatTensor(batch_tgt)
                if len(batch_regular_inds) > 0:
                    batch_regular_inds = torch.cat(batch_regular_inds, dim=0)
                    batch_regular_shapes = torch.cat(batch_regular_shapes, dim=0)
                else:
                    batch_regular_inds = None
                    batch_regular_shapes = None
                if len(batch_weighted_inds) > 0:
                    batch_weighted_inds = torch.cat(batch_weighted_inds, dim=0)
                    batch_weighted_shapes = torch.cat(batch_weighted_shapes, dim=0)
                    batch_weighted_kernels = torch.cat(batch_weighted_kernels, dim=0)
                    batch_weighted_bias = torch.cat(batch_weighted_bias, dim=0)
                else:
                    batch_weighted_inds = None
                    batch_weighted_shapes = None
                    batch_weighted_kernels = None
                    batch_weighted_bias = None
                batch = UniqueDict([
                    (DK_BATCH_SIZE, len(batch_list)),
                    (DK_BATCH_CG_REGULAR_IDX, batch_regular_inds),
                    (DK_BATCH_CG_REGULAR_SHAPES, batch_regular_shapes),
                    (DK_BATCH_CG_WEIGHTED_IDX, batch_weighted_inds),
                    (DK_BATCH_CG_WEIGHTED_SHAPES, batch_weighted_shapes),
                    (DK_BATCH_CG_WEIGHTED_KERNELS, batch_weighted_kernels),
                    (DK_BATCH_CG_WEIGHTED_BIAS, batch_weighted_bias),
                    (DK_BATCH_EDGE_TSR_LIST, batch_edge_list),
                    (DK_BATCH_LAST_NODE_IDX_LIST, batch_last_node_idx_list),
                    (DK_BATCH_UNIQUE_STR_ID_SET, batch_unique_str_id_set),
                    (DK_BATCH_TARGET_TSR, batch_tgt),
                ])
                if len(batch_unique_str_id_set) < len(batch_list):
                    print("Collected {} unique features but batch size is {}".format(len(batch_unique_str_id_set),
                                                                                     len(batch_list)))
                idx += self.batch_size
                self.batches.append(batch)
                if bar is not None: bar.update(1)
        if bar is not None: bar.close()
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.batches)

    def __iter__(self):
        self.shuffle()
        self.curr_batch_idx = 0
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        if self.curr_batch_idx >= len(self.batches):
            self.shuffle()
            self.curr_batch_idx = 0
            raise StopIteration()
        next_batch = self.batches[self.curr_batch_idx]
        self.curr_batch_idx += 1
        return next_batch

    def get_overlapping_data_count(self, loader):
        if not isinstance(loader, CGRegressDataLoader):
            print("Type mismatch, no overlaps by default")
            return 0
        n_unique_overlaps = 0
        my_data = set()
        for batch in self:
            batch_unique_str_id_set = batch[DK_BATCH_UNIQUE_STR_ID_SET]
            for str_id in batch_unique_str_id_set:
                my_data.add(str_id)
        for batch in loader:
            batch_unique_str_id_set = batch[DK_BATCH_UNIQUE_STR_ID_SET]
            for str_id in batch_unique_str_id_set:
                if str_id in my_data:
                    n_unique_overlaps += 1
        return n_unique_overlaps
