import copy
import hashlib
import collections
from model_src.comp_graph.tf_comp_graph import ComputeGraph
from utils.graph_utils import topo_sort_dfs, get_reverse_adj_dict, adj_dict_to_edge_list, edge_list_to_edge_pairs


def get_topo_sorted_nodes(nodes, src2dst_ids):
    """
    Note: sorted nodes will be the most dependent node first
    """
    new_src2dst_ids = collections.defaultdict(set)
    for k, v in src2dst_ids.items():
        new_src2dst_ids[k] = copy.deepcopy(v)
    id2node = {n.str_id:n for n in nodes}
    sorted_ids = topo_sort_dfs([n.str_id for n in nodes], new_src2dst_ids)
    sorted_nodes = [id2node[_id] for _id in sorted_ids]
    assert len(sorted_nodes) == len(nodes)
    return sorted_nodes


def get_simple_cg_str_id(cg:ComputeGraph, use_hash=True):
    """
    A quick and naive way to id a compute graph
    There is a limit on this id's ability to tell apart two CGs
    This limit should be at least as good as the limit of the gnn features to distinguish two CGs
    TODO: find a better way
    """
    nodes = cg.nodes
    src2dst_ids = cg.src_id2dst_ids_dict
    dst2src_ids = get_reverse_adj_dict(src2dst_ids)
    nodes = get_topo_sorted_nodes(nodes, dst2src_ids)
    edge_list = adj_dict_to_edge_list(src2dst_ids)
    edge_pairs = edge_list_to_edge_pairs(edge_list)
    cg_node_ids = []
    for ni, node in enumerate(nodes):
        if node.type_idx == 1: # Weighted node
            node_id = "<op{}res[{}]shape[{}]strides[{}]use_bias[{}]>".format(
                node.op_type_idx,
                ",".join([str(v) for v in node.resolution]),
                ",".join([str(v) for v in node.shape]),
                str(node.strides),
                str(node.metadata["use_bias"]) if node.metadata is not None and "use_bias" in node.metadata else "None")
        else:
            node_id = "<op{}res[{}]strides[{}]>".format(
                node.op_type_idx,
                ",".join([str(v) for v in node.resolution]),
                str(node.strides))
        cg_node_ids.append(node_id)
    cg_node_ids.sort()
    _id = "#".join(cg_node_ids) + "Edges:[{}]".format(edge_pairs)
    if use_hash:
        _id = hashlib.sha512(_id.encode("UTF-8")).hexdigest()
    return _id
