import torch
import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import datetime
import os

def plot_error_timeline(error_dates, epoch, patch, filename="error_timeline", start_date=None, end_date=None, figsize=(10, 2), save_dir="weki_JODIE"):
    
    filename = filename + "_epoch_" + str(epoch) + "_patch_" + str(patch) + ".png"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    full_path = os.path.join(save_dir, filename)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.scatter(error_dates, [0.5] * len(error_dates), color='red', s=50)
    
    ax.set_xlim([start_date if start_date else min(error_dates), end_date if end_date else max(error_dates)])
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.xaxis.set_minor_locator(plt.MaxNLocator(100))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(full_path, dpi=300)

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor, node_interact_times: np.ndarray, epoch=0, patch=0, if_print = False):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    time_interval = 100
    threshold_value = 0.02

    begin_time = min(node_interact_times)
    end_time = max(node_interact_times)
    begin_to_end = end_time - begin_time
    number_of_interval = int(begin_to_end/time_interval) + 1
    ada = np.zeros(number_of_interval)

    random_times = begin_time + (end_time - begin_time) * np.random.rand(node_interact_times.shape[0])
    node_interact_times = np.concatenate([node_interact_times, random_times])
    node_interact_times = np.sort(node_interact_times)
    
    predicts = predicts.cpu().detach().numpy()
    threshold = 0.5
    predicted_labels = (predicts > threshold).astype(int).squeeze()
    labels = labels.cpu().numpy()

    errors = predicted_labels ^ labels.astype(int)
    incorrect_times = node_interact_times[errors == 1]
    num_errors = np.sum(errors)

    # if if_print:
    #     plot_error_timeline(incorrect_times, start_date=begin_time, end_date=end_time, epoch=epoch, patch=patch)

    indices = ((node_interact_times - begin_time) / time_interval).astype(int)
    np.add.at(ada, indices, errors)
    ada /= time_interval
    ada = np.where(ada < threshold_value, 0, ada)
    ada = np.mean(ada)

    error_indices = np.where((errors == 1) & (np.arange(len(errors)) < len(node_interact_times)))[0]
    if len(error_indices) == 0:
        time_differences = np.array([0])
    else:
        shifted_indices = np.clip(error_indices - 1, 0, len(node_interact_times) - 1)
        time_differences = node_interact_times[error_indices] - node_interact_times[shifted_indices]

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    intensity = num_errors/begin_to_end
    aat = np.mean(time_differences)
    mat = min(time_differences)
    
    incorrect_times = (incorrect_times - begin_time) / (end_time - begin_time)
    
    if len(incorrect_times) == 0:
        mean_err = 0.5
        var_err = 0
    else:
        mean_err = np.mean(incorrect_times)
        var_err = np.var(incorrect_times)

    return {'average_precision': average_precision, 'roc_auc': roc_auc, 'intensity': intensity, 'ada': ada, 'aat': aat, 'mat': mat, 'mean_err': mean_err, 'var_err': var_err}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
