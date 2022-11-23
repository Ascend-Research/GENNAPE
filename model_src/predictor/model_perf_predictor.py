import time
import torch
import collections
from tqdm import tqdm
from constants import *
from utils.model_utils import device
from utils.eval_utils import get_regression_metrics, get_regression_rank_metrics


def train_predictor(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper, num_epochs,
                    max_gradient_norm=5.0, eval_start_epoch=1, eval_every_epoch=1,
                    rv_metric_name="mean_absolute_percent_error", completed_epochs=0,
                    dev_loader=None, checkpoint=True):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(num_epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_score = run_predictor_epoch(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper,
                                          rv_metric_name=rv_metric_name, max_grad_norm=max_gradient_norm,
                                          curr_epoch=report_epoch)
        book_keeper.log("Train score at epoch {}: {}".format(report_epoch, train_score))
        if checkpoint:
            book_keeper.checkpoint_model("_latest.pt", report_epoch, model, optimizer)

        if dev_loader is not None:
            with torch.no_grad():
                model.eval()
                if report_epoch >= eval_start_epoch and report_epoch % eval_every_epoch == 0:
                    dev_score = run_predictor_epoch(batch_fwd_func, model, dev_loader, criterion, None, book_keeper,
                                                    rv_metric_name=rv_metric_name, desc="Dev",
                                                    max_grad_norm=max_gradient_norm,
                                                    curr_epoch=report_epoch)
                    book_keeper.log("Dev score at epoch {}: {}".format(report_epoch, dev_score))
                    if checkpoint:
                        book_keeper.checkpoint_model("_best.pt", report_epoch, model, optimizer, eval_perf=dev_score)
                        book_keeper.report_curr_best()
        book_keeper.log("")


def run_predictor_epoch(batch_fwd_func, model, loader, criterion, optimizer, book_keeper,
                        desc="Train", curr_epoch=0, max_grad_norm=5.0, report_metrics=True,
                        rv_metric_name="mean_absolute_percent_error"):
    """
    Compatible with a predictor/loader that batches same-sized graphs
    """
    start = time.time()
    total_loss, n_instances = 0., 0
    metrics_dict = collections.defaultdict(float)
    preds, targets = [], []
    for batch in tqdm(loader, desc=desc, ascii=True):
        batch_vals = batch_fwd_func(model, batch)
        truth = batch[DK_BATCH_TARGET_TSR].to(device())
        pred = batch_vals.squeeze(1)
        loss = criterion(pred, truth)
        total_loss += loss.item() * batch[DK_BATCH_SIZE]
        preds.extend(pred.detach().tolist())
        targets.extend(truth.detach().tolist())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_instances += batch[DK_BATCH_SIZE]
    elapsed = time.time() - start
    rv_loss = total_loss / n_instances
    msg = desc + " epoch: {}, loss: {}, elapsed time: {}".format(curr_epoch, rv_loss, elapsed)
    book_keeper.log(msg)
    if report_metrics:
        metrics_dict = get_regression_metrics(preds, targets)
        rank_metrics = get_regression_rank_metrics(preds, targets,
                                                   top_overlap_k_list=(5, 10, 25, 50),
                                                   verbose=True)
        metrics_dict["spearman_rho"] = rank_metrics["spearman_rho"]
        metrics_dict["spearman_p"] = rank_metrics["spearman_p"]
        metrics_dict["top-5 overlaps"] = rank_metrics["top-5 overlaps"]
        metrics_dict["top-10 overlaps"] = rank_metrics["top-10 overlaps"]
        metrics_dict["top-25 overlaps"] = rank_metrics["top-25 overlaps"]
        metrics_dict["top-50 overlaps"] = rank_metrics["top-50 overlaps"]
        book_keeper.log("{} performance: {}".format(desc, str(metrics_dict)))
    return rv_loss if not report_metrics else metrics_dict[rv_metric_name]


def run_predictor_demo(batch_fwd_func, model, loader, log_f=print,
                       n_batches=1, normalize_constant=None,
                       input_str_key=None):
    n_visited = 0
    input_str_list = []
    preds, targets = [], []
    for batch in loader:
        if n_visited == n_batches:
            break
        batch_vals = batch_fwd_func(model, batch)
        truth = batch[DK_BATCH_TARGET_TSR].to(device())
        pred = batch_vals.squeeze(1)
        pred_list = pred.detach().tolist()
        target_list = truth.detach().tolist()
        preds.extend(pred_list)
        targets.extend(target_list)
        n_visited += 1
        if input_str_key is not None:
            batch_input_str = batch[input_str_key]
            for bi in range(len(pred_list)):
                input_str_list.append(batch_input_str[bi])
    for i, pred in enumerate(preds):
        input_str = input_str_list[i] if len(input_str_list) == len(preds) else None
        if input_str is not None:
            log_f("Input: {}".format(input_str))
        log_f("Pred raw: {}".format(pred))
        log_f("Truth raw: {}".format(targets[i]))
        if normalize_constant is not None:
            log_f("Normalize constant: {}".format(normalize_constant))
            log_f("Pred un-normalized: {:.3f}".format(pred * normalize_constant))
            log_f("Truth un-normalized: {:.3f}".format(targets[i] * normalize_constant))
        log_f("")
