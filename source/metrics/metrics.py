import torch
import torch.nn as nn

import torchmetrics
import torchmetrics.functional as F

from functools import reduce


class PredictionRejectionRatio(torchmetrics.Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()
        # uncertainty measures
        self.add_state('measures', default=[], dist_reduce_fx='cat')

        # indicators of incorrect prediction
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    def update(self, measures, targets):
        self.measures.append(measures)
        self.targets.append(targets)

    def compute(self):
        measures = torch.cat(self.measures, dim=0)
        targets = torch.cat(self.targets, dim=0)

        indices = torch.argsort(measures, descending=False)
        targets = targets[indices]

        num_samples = len(targets)
        num_misclassifications = torch.sum(targets).item()

        model_rejection_curve, percentage_space = [], []
        
        ### compute auc for model rejection curve
        # increase the top uncertainty threshold accepting more predictions
        # and check the predictive error of model & oracle system
        for i in range(1, num_samples + 1):
            model_rejection_curve.append(torch.sum(targets[:i]).item() / num_samples)
            percentage_space.append(i / num_samples)

        model_rejection_curve = torch.tensor(model_rejection_curve)
        percentage_space = torch.tensor(percentage_space)
        auc_model = F.auc(percentage_space, model_rejection_curve)
        
        ### compute auc for random rejection curve
        base_accuracy = model_rejection_curve[-1] # the same as num_misclassifications / num_samples
        random_rejection_curve = []
        
        for i in range(1, num_samples + 1):
            random_rejection_curve.append(i / num_samples * base_accuracy)

        random_rejection_curve = torch.tensor(random_rejection_curve)
        auc_random = F.auc(percentage_space, random_rejection_curve)

        ### compute auc for oracle rejection curve
        oracle_rejection_sub_curve = []
        
        for i in range(1, num_misclassifications + 1):
            oracle_rejection_sub_curve.append(i / num_misclassifications * base_accuracy)
        
        # oracle_rejection_curve = torch.zeros(num_samples)
        # oracle_rejection_curve[-num_misclassifications:] = torch.tensor(oracle_rejection_sub_curve)
        oracle_rejection_curve = torch.cat([
            torch.zeros(num_samples - num_misclassifications), 
            torch.tensor(oracle_rejection_sub_curve)
        ])

        auc_oracle = F.auc(percentage_space, oracle_rejection_curve)

        rejection_ratio = (auc_random - auc_model) / (auc_random - auc_oracle)
        return rejection_ratio


class AUPRC(torchmetrics.Metric):
    higher_is_better = False
    full_state_update = True

    def __init__(self):
        super().__init__()
        # uncertainty measures
        self.add_state('measures', default=[], dist_reduce_fx='cat')

        # indicators of incorrect prediction
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    def update(self, measures, targets):
        self.measures.append(measures)
        self.targets.append(targets)

    def compute(self):
        measures = torch.cat(self.measures, dim=0)
        targets = torch.cat(self.targets, dim=0)

        indices = torch.argsort(measures, descending=False)
        targets = targets[indices]

        num_samples = len(targets)
        model_rejection_curve, percentage_space = [], []
        
        ### compute auc for model rejection curve
        # increase the top uncertainty threshold accepting more predictions
        # and check the predictive error of model
        for i in range(1, num_samples + 1):
            model_rejection_curve.append(torch.sum(targets[:i]).item() / num_samples)
            percentage_space.append(i / num_samples)

        model_rejection_curve = torch.tensor(model_rejection_curve)
        percentage_space = torch.tensor(percentage_space)
        au_rejection_curve = F.auc(percentage_space, model_rejection_curve)
        
        return au_rejection_curve


metric_name_to_class = {
    'accuracy': torchmetrics.Accuracy,
    'auroc': torchmetrics.AUROC,
    'ap': torchmetrics.AveragePrecision,
    'prr': PredictionRejectionRatio,
    'auprc': AUPRC,
}


def create_metrics_per_metric_name(problem_config):
    return nn.ModuleDict({
        metric_name: metric_name_to_class[metric_name](**problem_config['metric_args']) 
        for metric_name in problem_config['metric_names']
    })


def create_metrics_per_problem_name(metrics_config):
    metrics_config = {} if metrics_config is None else metrics_config
    
    return nn.ModuleDict({
        problem_name: create_metrics_per_metric_name(problem_config) 
        for problem_name, problem_config in metrics_config.items()
    })


def create_misclassification_targets(predictions, labels, device):
    hards = torch.argmax(predictions, dim=1)
    targets = (hards != labels).type(torch.long).to(device)
    return targets


def create_ood_targets(masks, device):
    some_mask = next(iter(masks.values()))
    targets = torch.zeros_like(some_mask, dtype=torch.long, device=device)

    for mask_name, mask in masks.items():
        condition = 'out' in mask_name
        targets[mask] = int(condition)

    return targets


def prepare_masks_per_step(masks, step_name):
    return {mask_name: mask for mask_name, mask in masks.items() if step_name in mask_name}

def reduce_masks(masks, fn):
    return reduce(fn, masks.values())