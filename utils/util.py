import torch
import numpy as np


def float_list2str(float_list):
    return str(np.round(np.mean(float_list), 2)) + '+-' + str(
        np.round(np.std(float_list), 2))


def cal_grad_norm(model):
    gradient_norms = []
    grad_norm_display = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad)
            gradient_norms.append((name, grad_norm))
            grad_norm_display += grad_norm
    return grad_norm_display


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    return acc


from sklearn.metrics import roc_auc_score


def auc_score(output, labels):
    probs = torch.softmax(output, dim=1)
    probs = probs[:, 1].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    auc = roc_auc_score(labels, probs)
    return auc


def get_metric(output, labels, is_auc=False):
    if is_auc:
        return auc_score(output, labels)
    else:
        return accuracy(output, labels)
