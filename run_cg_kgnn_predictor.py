import time
import copy
import random
from params import *
import torch_geometric
import utils.model_utils as m_util
from model_src.demo_functions import *
from utils.misc_utils import RunningStatMeter
from model_src.model_helpers import BookKeeper
from model_src.comp_graph.tf_comp_graph import OP2I
from model_src.comp_graph.tf_comp_graph_models import make_cg_regressor
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from model_src.comp_graph.tf_comp_graph_dataloaders import CGRegressDataLoader
from utils.model_utils import set_random_seed, device, add_weight_decay, get_activ_by_name
from model_src.predictor.model_perf_predictor import train_predictor, run_predictor_demo


"""
Naive accuracy predictor training routine
For building a generalizable predictor interface
"""


def prepare_local_params(parser, ext_args=None):
    parser.add_argument("-model_name", required=False, type=str,
                        default="Demo")
    parser.add_argument("-family_train", required=False, type=str,
                        default="nb101"
                        )
    parser.add_argument('-family_test', required=False, type=str,
                        default="nb201c10#50"
                                "+nb301#50"
                                "+ofa_pn#50"
                                "+ofa_mbv3#50"
                                "+ofa_resnet#50"
                                "+hiaml#50"
                                "+inception#50"
                                "+two_path#50")
    parser.add_argument("-dev_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-test_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-epochs", required=False, type=int,
                        default=40)
    parser.add_argument("-fine_tune_epochs", required=False, type=int,
                        default=100)
    parser.add_argument("-batch_size", required=False, type=int,
                        default=32)
    parser.add_argument("-initial_lr", required=False, type=float,
                        default=0.0001)
    parser.add_argument("-in_channels", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-hidden_size", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-out_channels", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-num_layers", help="", type=int,
                        default=6, required=False)
    parser.add_argument("-dropout_prob", help="", type=float,
                        default=0.0, required=False)
    parser.add_argument("-aggr_method", required=False, type=str,
                        default="mean")
    parser.add_argument("-gnn_activ", required=False, type=str,
                        default="tanh")
    parser.add_argument("-reg_activ", required=False, type=str,
                        default=None)
    parser.add_argument('-gnn_type', required=False,
                        default="GraphConv")
    parser.add_argument("-normalize_HW_per_family", required=False, action="store_true",
                        default=False)
    parser.add_argument('-e_chk', type=str, default=None, required=False)
    return parser.parse_args(ext_args)


def get_family_train_size_dict(args):
    if args is None:
        return {}
    rv = {}
    for arg in args:
        if "#" in arg:
            fam, size = arg.split("#")
        else:
            fam = arg
            size = 0
        rv[fam] = int(float(size))
    return rv


def main(params):
    params.model_name = "gpi_acc_predictor_{}_seed{}".format(params.model_name, params.seed)
    book_keeper = BookKeeper(log_file_name=params.model_name + ".txt",
                             model_name=params.model_name,
                             saved_models_dir=params.saved_models_dir,
                             init_eval_perf=float("inf"), eval_perf_comp_func=lambda old, new: new < old,
                             saved_model_file=params.saved_model_file,
                             logs_dir=params.logs_dir)

    if type(params.family_test) is str:
        families_train = list(v for v in set(params.family_train.split("+")) if len(v) > 0)
        families_train.sort()
        families_test = params.family_test.split("+")
    else:
        families_train = params.family_train
        families_test = params.family_test

    book_keeper.log("Params: {}".format(params), verbose=False)
    set_random_seed(params.seed, log_f=book_keeper.log)
    book_keeper.log("Train Families: {}".format(families_train))
    book_keeper.log("Test Families: {}".format(families_test))

    families_test = get_family_train_size_dict(families_test)

    data_manager = FamilyDataManager(families_train, log_f=book_keeper.log)
    family2sets = \
        data_manager.get_regress_train_dev_test_sets(params.dev_ratio, params.test_ratio,
                                                     normalize_HW_per_family=params.normalize_HW_per_family,
                                                     normalize_target=False, group_by_family=True)

    train_data, dev_data, test_data = [], [], []
    for f, (fam_train, fam_dev, fam_test) in family2sets.items():
        train_data.extend(fam_train)
        dev_data.extend(fam_dev)
        test_data.extend(fam_test)

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    book_keeper.log("Train size: {}".format(len(train_data)))
    book_keeper.log("Dev size: {}".format(len(dev_data)))
    book_keeper.log("Test size: {}".format(len(test_data)))

    b_node_size_meter = RunningStatMeter()
    for g, _ in train_data + dev_data + test_data:
        b_node_size_meter.update(len(g))
    book_keeper.log("Max num nodes: {}".format(b_node_size_meter.max))
    book_keeper.log("Min num nodes: {}".format(b_node_size_meter.min))
    book_keeper.log("Avg num nodes: {}".format(b_node_size_meter.avg))

    train_loader = CGRegressDataLoader(params.batch_size, train_data)
    dev_loader = CGRegressDataLoader(params.batch_size, dev_data)
    test_loader = CGRegressDataLoader(params.batch_size, test_data)

    book_keeper.log(
        "{} overlap(s) between train/dev loaders".format(train_loader.get_overlapping_data_count(dev_loader)))
    book_keeper.log(
        "{} overlap(s) between train/test loaders".format(train_loader.get_overlapping_data_count(test_loader)))
    book_keeper.log(
        "{} overlap(s) between dev/test loaders".format(dev_loader.get_overlapping_data_count(test_loader)))

    book_keeper.log("Initializing {}".format(params.model_name))

    if "GINConv" in params.gnn_type:
        def gnn_constructor(in_channels, out_channels):
            nn = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels),
                                     torch.nn.Linear(in_channels, out_channels),
                                     )
            return torch_geometric.nn.GINConv(nn=nn)
    else:
        def gnn_constructor(in_channels, out_channels):
            return eval("torch_geometric.nn.%s(%d, %d)"
                        % (params.gnn_type, in_channels, out_channels))

    model = make_cg_regressor(n_unique_labels=len(OP2I().build_from_file()), out_embed_size=params.in_channels,
                              shape_embed_size=8, kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6,
                              hidden_size=params.hidden_size, out_channels=params.out_channels,
                              gnn_constructor=gnn_constructor,
                              gnn_activ=get_activ_by_name(params.gnn_activ), n_gnn_layers=params.num_layers,
                              dropout_prob=params.dropout_prob, aggr_method=params.aggr_method,
                              regressor_activ=get_activ_by_name(params.reg_activ)).to(device())

    if params.e_chk is not None:
        book_keeper.load_model_checkpoint(model, allow_silent_fail=False, skip_eval_perfs=True,
                                          checkpoint_file=params.e_chk)
        book_keeper.log("Loaded checkpoint: {}".format(params.e_chk))

    perf_criterion = torch.nn.MSELoss()
    model_params = add_weight_decay(model, weight_decay=0.)
    optimizer = torch.optim.Adam(model_params, lr=params.initial_lr)

    book_keeper.log(model)
    book_keeper.log("Model name: {}".format(params.model_name))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    book_keeper.log("Number of trainable parameters: {}".format(n_params))

    reg_metrics = ["MSE", "MAE", "MAPE"]
    kt_threshs = [0.0001]
    ndcg_ks = [50, 10]

    def _batch_fwd_func(_model, _batch):
        # Define how a batch is handled by the model
        regular_node_inds = _batch[DK_BATCH_CG_REGULAR_IDX]
        regular_node_shapes = _batch[DK_BATCH_CG_REGULAR_SHAPES]
        weighted_node_inds = _batch[DK_BATCH_CG_WEIGHTED_IDX]
        weighted_node_shapes = _batch[DK_BATCH_CG_WEIGHTED_SHAPES]
        weighted_node_kernels = _batch[DK_BATCH_CG_WEIGHTED_KERNELS]
        weighted_node_bias = _batch[DK_BATCH_CG_WEIGHTED_BIAS]
        edge_tsr_list = _batch[DK_BATCH_EDGE_TSR_LIST]
        batch_last_node_idx_list = _batch[DK_BATCH_LAST_NODE_IDX_LIST]
        return _model(regular_node_inds, regular_node_shapes,
                      weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                      edge_tsr_list, batch_last_node_idx_list)

    book_keeper.log("Training for {} epochs".format(params.epochs))
    start = time.time()
    try:
        train_predictor(_batch_fwd_func, model, train_loader, perf_criterion, optimizer, book_keeper,
                        num_epochs=params.epochs, max_gradient_norm=params.max_gradient_norm, dev_loader=dev_loader)
    except KeyboardInterrupt:
        book_keeper.log("Training interrupted")

    book_keeper.report_curr_best()
    book_keeper.load_model_checkpoint(model, allow_silent_fail=True, skip_eval_perfs=True,
                                      checkpoint_file=P_SEP.join([book_keeper.saved_models_dir,
                                                                  params.model_name + "_best.pt"]))
    end = time.time()
    with torch.no_grad():
        model.eval()
        book_keeper.log("===============Predictions===============")
        run_predictor_demo(_batch_fwd_func, model, test_loader,
                           n_batches=10, log_f=book_keeper.log)
        book_keeper.log("===============Overall Test===============")
        test_labels, test_preds = get_reg_truth_and_preds(model, test_loader, _batch_fwd_func)
        test_reg_metrics = pure_regressor_metrics(test_labels, test_preds)
        for i, metric in enumerate(reg_metrics):
            book_keeper.log("Test {}: {}".format(metric, test_reg_metrics[i]))

        overall_sp_rho = correlation_metrics(test_labels, test_preds)[0]
        book_keeper.log("Test Spearman Rho: {}".format(overall_sp_rho))

        test_kt_metrics = kendall_with_filts(test_labels, test_preds, filters=kt_threshs)
        for i, filt in enumerate(kt_threshs):
            book_keeper.log("Test KT{}: {}".format(filt, test_kt_metrics[i]))

        test_ndcg_metrics = ndcg(test_labels, test_preds, k_list=ndcg_ks)
        for i, k in enumerate(ndcg_ks):
            book_keeper.log("Test NDCG@{}: {}".format(k, test_ndcg_metrics[i]))

        book_keeper.log("Total time: %s" % (end - start))
        results_list = [overall_sp_rho]

    foreign_families = tuple(families_test.keys())
    book_keeper.log("Starting fine-tune on foreign families: {}".format(foreign_families))

    ff_manager = FamilyDataManager(families=foreign_families, log_f=book_keeper.log)
    ff_data = ff_manager.get_regress_train_dev_test_sets(0, 1.0,
                                                         group_by_family=True,
                                                         normalize_HW_per_family=params.normalize_HW_per_family,
                                                         normalize_target=False)

    for family, size in families_test.items():
        # For test families, merge test and validation partitions into one.
        book_keeper.log("Merging all data into test set for family {}".format(family))
        foreign_data = ff_data[family][-1] + ff_data[family][-2]

        foreign_data.sort(key=lambda x: x[1], reverse=True)

        ft_shuffle_inds = list(range(len(foreign_data)))
        np.random.seed(params.seed)
        np.random.shuffle(ft_shuffle_inds)
        foreign_data = [foreign_data[i] for i in ft_shuffle_inds]
        fine_tune_data = foreign_data[:size]
        foreign_test_data = foreign_data[size:]

        book_keeper.log("Foreign family {} fine-tune size: {}".format(family, len(fine_tune_data)))
        book_keeper.log("Foreign family {} test size: {}".format(family, len(foreign_test_data)))
        foreign_test_loader = CGRegressDataLoader(1, foreign_test_data)

        if len(fine_tune_data) > 0:

            ft_model = copy.deepcopy(model)
            ft_opt = torch.optim.Adam(ft_model.parameters(), lr=params.initial_lr)

            ft_loader = CGRegressDataLoader(1, fine_tune_data)

            book_keeper.log("Fine-tuning for {} epochs".format(params.fine_tune_epochs))
            train_predictor(_batch_fwd_func, ft_model, ft_loader, perf_criterion, ft_opt, book_keeper,
                            num_epochs=params.fine_tune_epochs, max_gradient_norm=params.max_gradient_norm,
                            dev_loader=None, checkpoint=False)

        with torch.no_grad():
            model.eval()
            foreign_labels, foreign_preds = get_reg_truth_and_preds(model, foreign_test_loader, _batch_fwd_func)
            test_reg_metrics = pure_regressor_metrics(foreign_labels, foreign_preds)
            for i, metric in enumerate(reg_metrics):
                book_keeper.log("{}-NoFT {}: {}".format(family, metric, test_reg_metrics[i]))

            no_ft_sp = correlation_metrics(foreign_labels, foreign_preds)[0]
            book_keeper.log("{}-NoFT Spearman Rho: {}".format(family, no_ft_sp))

            test_kt_metrics = kendall_with_filts(foreign_labels, foreign_preds, filters=kt_threshs)
            for i, filt in enumerate(kt_threshs):
                book_keeper.log("{}-NoFT KT{}: {}".format(family, filt, test_kt_metrics[i]))

            test_ndcg_metrics = ndcg(foreign_labels, foreign_preds, k_list=ndcg_ks)
            for i, k in enumerate(ndcg_ks):
                book_keeper.log("{}-NoFT NDCG@{}: {}".format(family, k, test_ndcg_metrics[i]))

            book_keeper.log("Total time: %s" % (end - start))

            results_list.append(no_ft_sp)

            if len(fine_tune_data) > 0:
                book_keeper.checkpoint_model("_{}_ft.pt".format(family), params.fine_tune_epochs,
                                             ft_model, ft_opt)

                foreign_labels, foreign_preds = get_reg_truth_and_preds(ft_model, foreign_test_loader, _batch_fwd_func)
                test_reg_metrics = pure_regressor_metrics(foreign_labels, foreign_preds)
                for i, metric in enumerate(reg_metrics):
                    book_keeper.log("{}-FT {}: {}".format(family, metric, test_reg_metrics[i]))

                ft_sp = correlation_metrics(foreign_labels, foreign_preds)[0]
                book_keeper.log("{}-FT Spearman Rho: {}".format(family, ft_sp))

                test_kt_metrics = kendall_with_filts(foreign_labels, foreign_preds, filters=kt_threshs)
                for i, filt in enumerate(kt_threshs):
                    book_keeper.log("{}-FT KT{}: {}".format(family, filt, test_kt_metrics[i]))

                test_ndcg_metrics = ndcg(foreign_labels, foreign_preds, k_list=ndcg_ks)
                for i, k in enumerate(ndcg_ks):
                    book_keeper.log("{}-FT NDCG@{}: {}".format(family, k, test_ndcg_metrics[i]))

                results_list.append(ft_sp)


if __name__ == "__main__":
    _parser = prepare_global_params()
    _args = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = _args.device_str
    main(_args)
    print("done")
