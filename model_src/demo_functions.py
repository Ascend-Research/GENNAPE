import torch
import numpy as np
from constants import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, ndcg_score
from scipy.stats import pearsonr, spearmanr


def get_reg_truth_and_preds(model, loader, fwd_func):

    labels = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch_labs = batch[DK_BATCH_TARGET_TSR]
            labels += batch_labs.detach().tolist()
            batch_preds = fwd_func(model, batch)
            preds += batch_preds.detach().tolist()

    labels = np.array(labels).squeeze()
    preds = np.array(preds).squeeze()

    return labels, preds


# Mean Squared Error - MSE
# Mean Absolute Error - MAE
# Mean Absolute Percentage Error - MAPE
def pure_regressor_metrics(targets, preds):

    # MSE, MAE, MAPE
    target_mse = mean_squared_error(targets, preds)
    target_mae = mean_absolute_error(targets, preds)
    target_mape = mean_absolute_percentage_error(targets, preds)

    return [target_mse, target_mae, target_mape]


# Spearmann Rank Correlation - SRCC
# Pearson Correlation
def correlation_metrics(targets, preds, pearson=False):

    metrics = []
    if pearson:
        # Pearson Correlation
        pcc, pp = pearsonr(targets, preds)
        metrics.append(pcc)

    # Spearman Correlation
    srcc, sp = spearmanr(targets, preds)
    metrics.append(srcc)

    return metrics


# Kendall's Tau - KT
def kendall_with_filts(targets, preds, filters=[0]):

    kts = []
    for filter_val in filters:
        filt_kt = pairwise_kt(targets, preds, filter=filter_val)
        kts.append(filt_kt)

    return kts


# NDCG
def ndcg(targets, preds, k_list=[-1]):

    rel_cal = RelevanceCalculator.from_data(targets, 20.)

    if len(targets.shape) < 2:
        targets = np.expand_dims(targets, axis=0)
    if len(preds.shape) < 2:
        preds = np.expand_dims(preds, axis=0)

    ndcgs = []
    for k in k_list:
        if k < 1:
            k = np.max(targets.shape)
        ndcg = ndcg_score(rel_cal(targets), preds, k=k, ignore_ties=False)
        ndcgs.append(ndcg)

    return ndcgs


def pairwise_kt(targets, preds, filter=0):

    pm_count, num_actual_pairs = 0, 0

    num_adj_coef = 1e6
    filter *= num_adj_coef
    targets = np.round(targets * num_adj_coef, 0)
    preds = np.round(preds * num_adj_coef, 0)
    idx = np.argsort(-targets)
    preds = preds[idx]
    targets = targets[idx]

    num_data = targets.shape[0]

    for i in range(num_data - 1):

        input2_accs = targets[i + 1:]
        input1_accs = targets[i]

        prediction = (preds[i] >= preds[i + 1:]) * 1.0

        prediction = 2 * prediction - 1
        if filter:
            filtered_pred_idx = (input1_accs - input2_accs) >= filter
            prediction = prediction * filtered_pred_idx

        pm_count += prediction.sum()

        num_actual_pairs += (np.abs(prediction) > 0).sum()

    kt = pm_count * 1. / num_actual_pairs
    return np.round(kt, 4)


# File copied from https://github.com/ultmaster/AceNAS/blob/main/gcn/benchmarks/metrics.py
class RelevanceCalculator:
    def __init__(self, lower, upper, scale):
        print(f'Creating relevance calculator: lower = {lower:.6f}, upper = {upper:.6f}, scale = {scale:.2f}', __name__)
        self.lower = lower
        self.upper = upper
        self.scale = scale

    @classmethod
    def from_data(cls, x, scale):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        lower = np.min(x)
        upper = np.max(x)
        return cls(lower, upper, scale)

    def __call__(self, x):
        if torch.is_tensor(x):
            return torch.clamp((x - self.lower) / (self.upper - self.lower), 0, 1) * self.scale
        else:
            return np.clip((x - self.lower) / (self.upper - self.lower), 0, 1) * self.scale