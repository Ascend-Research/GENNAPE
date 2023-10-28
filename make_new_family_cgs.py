import argparse
import pickle
from model_src.search_space.hiaml_two_path.arch_gen import sample_hiaml_net_config, sample_two_path_net_config
from model_src.search_space.hiaml_two_path.arch_gen import build_hiaml_cg, build_two_path_cg
from model_src.search_space.inception.arch_gen import sample_net_config as sample_inception_net_config
from model_src.search_space.inception.arch_gen import build_cg as build_inception_cg


"""
Script for generating CGs for the HiAML, Inception and Two-Path CGs. 
The souce code for these models can be found in:
- model_src.search_space.hiaml_two_path
- model_src.search_space.inception
NOTE: We did not fully exhaust these search spaces when generating the CG datasets in this paper.
- E.g., our HiAML cache contains, 4.6k architectures
    however, the total size is 14 blks ^ 4 stages = 38k total architectures, >2x the size of NB-201. 
    Inception and Two-Path are even larger.
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-family", type=str, default="hiaml", choices=["hiaml", "inception", "two_path"])
    parser.add_argument("-num_cgs", type=int, default=10)
    parser.add_argument("-out_cache_name", type=str, default=None)
    parser.add_argument("-exhaust", action="store_true", default=False)

    params = parser.parse_args()

    if params.exhaust:
        assert params.family == "hiaml"
        from itertools import product
        from model_src.search_space.hiaml_two_path.constants import HIAML_OPS
        params.out_cache_name = "hiaml_full_unlabeled_comp_graph_cache"
        stage_blk_list = [HIAML_OPS, HIAML_OPS, HIAML_OPS, HIAML_OPS]
        exhaust_hiaml_configs = product(*stage_blk_list)
        cg_list = []
        for config in exhaust_hiaml_configs:
            name = "_" + "_".join(x.split("_")[-1] for x in config)
            cg_list.append({'compute graph': build_hiaml_cg(config, name, H=32, W=32),
                            'original config': config})

    else:
        cg_list = []
        sample_func = eval(f"sample_{params.family}_net_config")
        maker_func = eval(f"build_{params.family}_cg")
        for i in range(params.num_cgs):
            cfg = sample_func()
            print("===")
            print(cfg)
            cg_list.append({'compute graph': maker_func(cfg, str(i), H=32, W=32),
                            'original config': cfg})

    if params.out_cache_name is None:
        params.out_cache_name = f"{params.family}_{params.num_cgs}_comp_graph_cache"

    with open(params.out_cache_name + ".pkl", "wb") as f:
        pickle.dump(cg_list, f, protocol=4)
