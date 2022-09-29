import torch
import numpy as np

import dgl
import pytorch_lightning as pl

import networkx as nx
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import DBSCAN

from .utils import dataset_name_to_prepare_graph_fn, dataset_name_to_process_graph_fn, dataset_name_to_prepare_masks_fn


class StandardGraphDataset():
    def __init__(self, dataset_config):
        super().__init__()
        self.config = dataset_config
        
        prepare_graph = dataset_name_to_prepare_graph_fn[self.config.dataset_name]
        graph = prepare_graph(self.config.default_dir, self.config.dataset_name, self.config.prepare_config)
        
        process_graph = dataset_name_to_process_graph_fn[self.config.dataset_name]
        graph = process_graph(graph, self.config.process_config)
        self.graph = graph
        
        prepare_masks = dataset_name_to_prepare_masks_fn[self.config.dataset_name]
        self.masks = prepare_masks(self.config.default_dir, self.config.dataset_name, self.config.split_config)
        # self.prepare_split_indices()


def get_dataset_class(dataset_class_name):
    dataset_name_to_class = {
        'standard': StandardGraphDataset,
        '': None,
    }
    
    return dataset_name_to_class[dataset_class_name]
