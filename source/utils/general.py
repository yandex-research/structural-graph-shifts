import random
import numpy as np
import torch
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(64)

def average_dataframe(something):
    return pd.DataFrame(something.mean(axis=0)).T


def to_dataframe(something):
    return pd.DataFrame([something]) if isinstance(something, dict) else pd.DataFrame(something)


def save_results(results, path):
    results.to_csv(path, index=False)


def save_history_results(history, history_experiment_path):
    path = f"{history_experiment_path}/results.csv"
    results = average_dataframe(to_dataframe(history))
    save_results(results, path)
    print(f"Results of inference procedure are saved in {path}")


def save_separate_results(metrics, separate_experiment_path):
    path = f"{separate_experiment_path}/results.csv"
    results = to_dataframe(metrics)
    save_results(results, path)
    print(f"Results of separate experiment are saved in {path}")