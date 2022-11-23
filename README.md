# Towards Generalized Neural Architecture Performance Estimators

This repository accompanies the paper

> GENNAPE: Towards Generalized Neural Architecture Performance Estimators\
> Keith G. Mills, Fred X. Han, Jialin Zhang, Fabian Chudak, Ali Safari Mamaghani, Mohammad Salameh, Wei Lu, 
> Shangling Jui and Di Niu\
> AAAI-23


We provide sample data for our Computation Graphs (CG), the API for loading and visualizing CGs, as well as a sample 
demo code for the *k*-GNN predictor.

### Main Dependencies
```
python==3.7
torch==1.8.1
torch_geometric==1.7.0
tensorflow==1.15.0
graphviz
```
See `requirements.txt` for more specifics

### Setting Up The Repository
1. Download and unpack CG_data.zip from the [public google drive folder](https://drive.google.com/drive/folders/1nTj8g6XbIU_PYvOBaXjylBHmgo-3okra)
2. Copy the `cache` and `data` folders to the top-level of this directory.
    * `cache` contains the CG files for the NAS-Bench (101, 201 and 301) as well as OFA (PN, MBv3 and ResNet-50) families.
    * `data` contains the CG files for the HiAML, Inception and Two-Path families.

### Running the Demo *k*-GNN predictor code
Use the top-level script `run_cg_kgnn_predictor.py`; notable parameters include:
* `-model_name`: String for the name of an experiment.
  * Log file will save under `logs/gpi_acc_predictor_{model_name}_seed{seed}.txt`
  * Model checkpoints will be saved under `/saved_models/gpi_acc_predictor_{model_name}_seed{seed}_{best, last}.pt`
* `-family_train`: Training family; default is NB-101
* `-family_test`: Testing/fine-tuning family. Default is all test families.
  * The flag `#50` denotes that 50 samples will be taken-out to fine-tune a model for epochs determined by `-fine_tune_epochs`
  * If this flag is not specified, e.g., `-family_test hiaml` is used, no fine-tuning will be performed.
* `-e_chk`: Allows user to load a specified model checkpoint.
* `-seed`: Random seed.
* `-gnn_type`: Default is 'GraphConv' (*k*-GNN), but can be changed to others, e.g., GCNConv, GINConv, etc.

### Visualization of CGs
* API function is `gviz_visualize`, found in `/model_src/comp_graph/tf_comp_graph.py`
* We provide a simple script, `visualize_cgs.py`, to print out N random architectures from a specified family to a target directory.
* The loaded figure shows the node with node features, e.g., kernel sizes, as well as input/output tensor shapes.

### Generation of new CGs from TensorFlow protobuf `.pb` families
* See `make_cg.py` for an example.
* We provide sample `.pb` files for EfficientNet-b0 and ResNet18 on the [public google drive folder](https://drive.google.com/drive/folders/1nTj8g6XbIU_PYvOBaXjylBHmgo-3okra). You can use this script to make CGs and print pictures of them, then compare to what the `.pb` of the model looks like in [Netron](https://netron.app/).

### Miscellaneous:
* The CG API is defined in `/model_src/comp_graph/tf_comp_graph.py`
* To access CGs directly, `/model_src/predictor/gpi_family_data_manager.py` handles loading CGs from a cache.