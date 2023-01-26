import sys
sys.path.append('..')

import os
import shutil
import argparse

from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
import torch

from fantasy.data import dataset as D
from fantasy.utils import config as C

import graph_tool as gt
from graph_tool import clustering, centrality, spectral, topology, draw, stats, generation


@dataclass
class SplitConfiguration(C.BaseConfig):
    data_root: str = '../datasets'
    config_root: str = '../configs'
    
    dataset_name: str = 'pubmed'
    strategy_name: str = 'random'
    version_name: str = 'extra'
    save: bool = True
    
    in_size: float = 0.5
    out_size: float = 0.5

    in_train_size: float = 0.3
    in_valid_size: float = 0.1
    out_valid_size: float = 0.1

def prepare_split_config(split_config_path):
    return SplitConfiguration(**C.read_config(split_config_path))



dataset_names = [
    'cora-ml',
    'citeseer',
    'pubmed',

    'amazon-computer',
    'amazon-photo',

    'coauthor-cs',
    'coauthor-physics',

    'roman-empire',
    'amazon-ratings',
    'workers',
    'questions',
]

strategy_names = [
    'random',
    'pagerank',
    'personalised',
    'clustering',
]



def prepare_dgl_graph(split_config):
    config_path = f'{split_config.config_root}/dataset_configs/{split_config.dataset_name}/random/sanity_dataset_config.yaml'

    dataset_config = C.DatasetConfiguration(**C.read_config(config_path))
    dataset_config.default_dir = split_config.data_root

    dataset = D.StandardGraphDataset(dataset_config)
    dataset.prepare_graph()

    graph = dataset.graph
    return graph

def prepare_graphtool_graph(graph_dgl):
    edge_list = torch.stack(graph_dgl.edges()).numpy().T

    graph_gt = gt.Graph()
    graph_gt.add_edge_list(edge_list)

    graph_gt.set_directed(False)
    stats.remove_parallel_edges(graph_gt)

    return graph_gt



def get_random_property_values(graph_gt, sign=1.0):
    property_values = sign * np.random.permutation(graph_gt.num_vertices())
    return property_values

def get_pagerank_property_values(graph_gt, sign=-1.0):
    property_values = sign * np.array(centrality.pagerank(graph_gt).get_array())
    return property_values

def get_personalised_property_values(graph_gt, sign=-1.0):
    pagerank_values = np.array(centrality.pagerank(graph_gt).get_array())

    ohe_mask = np.zeros_like(pagerank_values)
    ohe_mask[np.argmax(pagerank_values)] = 1.0

    property_gt = graph_gt.new_vertex_property('double')
    property_gt.a = ohe_mask

    property_values = sign * np.array(centrality.pagerank(graph_gt, pers=property_gt).get_array())
    return property_values

def get_clustering_property_values(graph_gt, sign=-1.0):
    property_values = sign * np.array(clustering.local_clustering(graph_gt).get_array())
    return property_values

strategy_name_to_property_values_fn = {
    'random': get_random_property_values,
    'pagerank': get_pagerank_property_values,
    'personalised': get_personalised_property_values,
    'clustering': get_clustering_property_values,
}



def prepare_split_masks_by_property(property_values, split_params):
    num_nodes = len(property_values)
    
    in_size = int(split_params['in_size'] * num_nodes)
    out_size = num_nodes - in_size

    in_train_size = int(split_params['in_train_size'] * num_nodes)
    in_valid_size = int(split_params['in_valid_size'] * num_nodes)
    in_test_size = in_size - in_train_size - in_valid_size

    out_valid_size = int(split_params['out_valid_size'] * num_nodes)
    out_test_size = out_size - out_valid_size

    cuts = np.cumsum([
        in_train_size, 
        in_valid_size, 
        in_test_size, 
        
        out_valid_size, 
        out_test_size
    ])

    argsort = np.argsort(property_values)
    node_ids = np.arange(num_nodes)

    node_ids_sorted = node_ids[argsort]
    node_ids_sorted[:in_size] = np.random.permutation(node_ids_sorted[:in_size])

    in_train_ids = node_ids_sorted[:cuts[0]]
    in_valid_ids = node_ids_sorted[cuts[0]:cuts[1]]
    in_test_ids = node_ids_sorted[cuts[1]:cuts[2]]

    out_valid_ids = node_ids_sorted[cuts[2]:cuts[3]]
    out_test_ids = node_ids_sorted[cuts[3]:cuts[4]]

    in_train_mask, in_valid_mask, in_test_mask, out_valid_mask, out_test_mask = [
        torch.zeros(num_nodes, dtype=torch.bool) for _ in range(5)
    ]
    in_train_mask[in_train_ids] = True
    in_valid_mask[in_valid_ids] = True
    in_test_mask[in_test_ids] = True

    out_valid_mask[out_valid_ids] = True
    out_test_mask[out_test_ids] = True

    return {
        'train': in_train_mask,
        'valid_in': in_valid_mask,
        'test_in': in_test_mask,

        'valid_out': out_valid_mask,
        'test_out': out_test_mask,
    }

def prepare_split_params(split_config):
    return {
        'in_size': split_config.in_size,
        'in_train_size': split_config.in_train_size,
        'in_valid_size': split_config.in_valid_size,

        'out_size': split_config.out_size,
        'out_valid_size': split_config.out_valid_size,
    }


def prepare_split_masks(graph_gt, split_params, split_config):
    get_property_values_fn = strategy_name_to_property_values_fn[split_config.strategy_name]
    property_values = get_property_values_fn(graph_gt)

    split_masks = prepare_split_masks_by_property(property_values, split_params)
    split_path = f'{split_config.data_root}/{split_config.dataset_name}/splits/type={split_config.strategy_name}-{split_config.version_name}.pth'
    
    if split_config.save: torch.save(split_masks, split_path)
    return split_masks, property_values