import numpy as np
from scipy import io

import torch
import dgl


node_names = {
    'fraud-amazon': 'user'
}


def load_homogenous_graph(dataset_root, dataset_name, prepare_config):
    dataset_path = f"{dataset_root}/{dataset_name}/data.mat"
    data = io.loadmat(dataset_path)

    features = data['features'] if type(data['features']) == np.ndarray else data['features'].todense()
    labels = data['labels'].squeeze()

    rows, cols = data['edges']
    edge_index = torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)
    
    graph = dgl.graph(edge_index, num_nodes=len(features))
    graph.ndata['features'] = torch.tensor(features, dtype=torch.float)
    graph.ndata['labels'] = torch.tensor(labels, dtype=torch.long)
    
    return graph


dataset_name_to_load_graph_fn = {
    'amazon-computer': load_homogenous_graph,
    'amazon-photo': load_homogenous_graph,
    'coauthor-cs': load_homogenous_graph,
    'coauthor-physics': load_homogenous_graph,
    'cora-ml': load_homogenous_graph,
    'citeseer': load_homogenous_graph,
    'pubmed': load_homogenous_graph,
    'ogbn-products': load_homogenous_graph,
}


### process functions


def process_homogenous_graph(graph, process_config):
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    
    return graph


dataset_name_to_process_graph_fn = {
    'amazon-computer': process_homogenous_graph,
    'amazon-photo': process_homogenous_graph,
    'coauthor-cs': process_homogenous_graph,
    'coauthor-physics': process_homogenous_graph,
    'cora-ml': process_homogenous_graph,
    'citeseer': process_homogenous_graph,
    'pubmed': process_homogenous_graph,
    'ogbn-products': process_homogenous_graph,
}


### masks functions


def prepare_homogenous_masks(dataset_root, dataset_name, split_config):
    split_path = f"{dataset_root}/{dataset_name}/splits/{split_config['type']}.pth"
    masks = torch.load(split_path)
    return masks


dataset_name_to_prepare_masks_fn = {
    'amazon-computer': prepare_homogenous_masks,
    'amazon-photo': prepare_homogenous_masks,
    'coauthor-cs': prepare_homogenous_masks,
    'coauthor-physics': prepare_homogenous_masks,
    'cora-ml': prepare_homogenous_masks,
    'citeseer': prepare_homogenous_masks,
    'pubmed': prepare_homogenous_masks,
    'ogbn-products': prepare_homogenous_masks,
}