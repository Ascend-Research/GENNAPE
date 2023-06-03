from model_src.predictor.gpi_family_data_manager import _build_cache_nb201
from model_src.predictor.gpi_family_data_manager import _build_cache_nb101
from model_src.predictor.gpi_family_data_manager import _build_cache_nb301
from model_src.predictor.gpi_family_data_manager import _build_cache_ofa_pn_mbv3
from model_src.predictor.gpi_family_data_manager import _build_cache_ofa_resnet


"""
Simple Script to demonstrate how we generate the CGs in our caches from TensorFlow models.
For NAS-Bench and OFA families
Each cache, /cache/gpi_{family}_comp_graph_cache.pkl is a list of dictionaries
Each dictionary contains a CG, accuracy, flops, original config information, etc.
The original config is needed to generate the Computational Graph as a TensorFlow model
E.g., for NB-301, it is the DARTS Genotype.
This allows users to generate CGs for networks beyond what we provided (chosen at random) in our caches.

For a desired family, lookup the corresponding function in /model_src/predictor/gpi_family_data_manager.py and follow
the process and note the required dependencies from /model_src/search_space/
"""


if __name__ == "__main__":
    # 101
    _build_cache_nb101("cache/demo_nb101_comp_graph_cache.pkl")

    # 201
    _build_cache_nb201("cache/demo_nb201c10_comp_graph_cache.pkl")

    # 301
    _build_cache_nb301("cache/demo_nb301_comp_graph_cache.pkl")

    # OFA-MB
    _build_cache_ofa_pn_mbv3("cache/demo_ofa_comp_graph_cache.pkl")

    # OFA-RN
    _build_cache_ofa_resnet("cache/demo_ofa_resnet_comp_graph_cache.pkl")
