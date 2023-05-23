import sys
sys.path.append('..')

import os
import shutil
import argparse

from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
import torch

from source.data import dataset as D
from source.utils import config as C

import networkx as nx
import networkx.algorithms as A



dataset_names = [
    'cora-ml',
    'citeseer',
    'pubmed',

    'amazon-computer',
    'amazon-photo',

    'coauthor-cs',
    'coauthor-physics',
]

strategy_names = [
    'random',
    'popularity',
    'locality',
    'density',
]


@dataclass
class SplitConfiguration(C.BaseConfig):
    data_root: str = '../datasets'
    config_root: str = '../configs'
    
    dataset_name: str = 'amazon-computer'
    property_name: str = 'popularity'
    save: bool = True
    
    in_train_size: float = 0.3
    in_valid_size: float = 0.1
    in_test_size: float = 0.1
    out_valid_size: float = 0.1
    out_test_size: float = 0.4


def prepare_split_config(split_config_path):
    return SplitConfiguration(**C.read_config(split_config_path))


def prepare_part_ratios(split_config):
    return [
        split_config.in_train_size,
        split_config.in_valid_size,
        split_config.in_test_size,
        split_config.out_valid_size,
        split_config.out_test_size,
    ]


def prepare_graph(split_config):
    config_path = f'{split_config.config_root}/dataset_configs/{split_config.dataset_name}/random/standard_dataset_config.yaml'

    dataset_config = C.DatasetConfiguration(**C.read_config(config_path))
    dataset_config.default_dir = split_config.data_root

    dataset = D.StandardGraphDataset(dataset_config)
    dataset.prepare_graph()

    graph_dgl = dataset.graph
    return graph_dgl


def prepare_split(graph_dgl, part_ratios, split_config, ascending=True):
    graph_nx = nx.Graph(graph_dgl.to_networkx())

    compute_property_fn = _property_name_to_compute_fn[split_config.property_name]
    property_values = compute_property_fn(graph_nx, ascending)

    node_masks = mask_nodes_by_property(property_values, part_ratios)
    node_masks_path = f'{split_config.data_root}/{split_config.dataset_name}/splits/{split_config.property_name}-bruh.pth'
    property_values_path = f'{split_config.data_root}/{split_config.dataset_name}/splits/{split_config.property_name}-bruh-values.npy'
    
    if split_config.save:
        torch.save(node_masks, node_masks_path)
        np.save(property_values_path, property_values)

    return node_masks, property_values


def mask_nodes_by_property(property_values, part_ratios):
    num_nodes = len(property_values)
    part_sizes = np.round(num_nodes * np.array(part_ratios)).astype(int)
    part_sizes[-1] -= np.sum(part_sizes) - num_nodes

    permutation = np.random.permutation(num_nodes)

    node_indices = np.arange(num_nodes)[permutation]
    property_values = property_values[permutation]
    in_distribution_size = np.sum(part_sizes[:3])

    node_indices_ordered = node_indices[np.argsort(property_values)]
    node_indices_ordered[:in_distribution_size] = np.random.permutation(
        node_indices_ordered[:in_distribution_size]
    )

    sections = np.cumsum(part_sizes)
    node_split = np.split(node_indices_ordered, sections)[:-1]
    mask_names = [
        "train",
        "valid_in",
        "test_in",
        "valid_out",
        "test_out",
    ]
    split_masks = {}

    for mask_name, node_indices in zip(mask_names, node_split):
        split_mask = np.zeros(num_nodes, dtype=bool)
        split_mask[node_indices] = True
        split_masks[mask_name] = torch.tensor(split_mask, dtype=bool)

    return split_masks


def _compute_random_property(graph_nx, ascending=True):
    property_values = np.random.permutation(graph_nx.number_of_nodes())
    return property_values


def _compute_popularity_property(graph_nx, ascending=True):
    direction = -1 if ascending else 1
    property_values = direction * np.array(list(A.pagerank(graph_nx).values()))
    return property_values


def _compute_locality_property(graph_nx, ascending=True):
    num_nodes = graph_nx.number_of_nodes()
    pagerank_values = np.array(list(A.pagerank(graph_nx).values()))

    personalization = dict(zip(range(num_nodes), [0.0] * num_nodes))
    personalization[np.argmax(pagerank_values)] = 1.0

    direction = -1 if ascending else 1
    property_values = direction * np.array(
        list(A.pagerank(graph_nx, personalization=personalization).values())
    )
    return property_values


def _compute_density_property(graph_nx, ascending=True):
    direction = -1 if ascending else 1
    property_values = direction * np.array(
        list(A.clustering(graph_nx).values())
    )
    return property_values


_property_name_to_compute_fn = {
    "random": _compute_random_property,
    "popularity": _compute_popularity_property,
    "locality": _compute_locality_property,
    "density": _compute_density_property,
}