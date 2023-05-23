import torch
import numpy as np

import dgl
import pytorch_lightning as pl

import networkx as nx
from collections import Counter


from .utils import (
    dataset_name_to_load_graph_fn, 
    dataset_name_to_process_graph_fn, 
    dataset_name_to_prepare_masks_fn
)


class StandardGraphDataset():
    def __init__(self, dataset_config):
        super().__init__()
        self.config = dataset_config

        self.graph = None
        self.masks = None
        
    def prepare_graph(self):
        load_graph_fn = dataset_name_to_load_graph_fn[self.config.dataset_name]
        graph = load_graph_fn(self.config.default_dir, self.config.dataset_name, self.config.prepare_config)
        
        process_graph_fn = dataset_name_to_process_graph_fn[self.config.dataset_name]
        graph = process_graph_fn(graph, self.config.process_config)
        
        self.graph = graph

    def prepare_masks(self):
        prepare_masks_fn = dataset_name_to_prepare_masks_fn[self.config.dataset_name]
        self.masks = prepare_masks_fn(self.config.default_dir, self.config.dataset_name, self.config.split_config)


def get_dataset_class(dataset_class_name):
    dataset_name_to_class = {
        'standard': StandardGraphDataset,
        '': None,
    }
    
    return dataset_name_to_class[dataset_class_name]
