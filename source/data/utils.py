import numpy as np
from scipy import io

import torch
import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder


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

    'roman-empire': load_homogenous_graph,
    'amazon-ratings': load_homogenous_graph,
    'minesweeper': load_homogenous_graph,
    'workers': load_homogenous_graph,
    'questions': load_homogenous_graph,
}


### process functions

NUM_ANCHOR_NODES = 5
NUM_CATEGORIES = 10


def prepare_degree_features(graph):
    degree_values = graph.in_degrees().numpy()
    
    unique_categories = [np.arange(NUM_CATEGORIES)]
    bins = np.r_[np.array([0]), np.logspace(0, NUM_CATEGORIES - 2, num=NUM_CATEGORIES - 1, base=2.0), np.array([np.inf])]
    category_values = (np.digitize(degree_values, bins) - 1).reshape(-1, 1)
    
    degree_features = OneHotEncoder(categories=unique_categories, sparse=False).fit_transform(category_values)
    degree_features = np.flip(np.cumsum(np.flip(degree_features, axis=1), axis=1), axis=1)
    return torch.tensor(degree_features.copy(), dtype=torch.float)


def prepare_distance_features(graph):
    graph_nx = nx.Graph(graph.to_networkx())
    num_nodes = graph.num_nodes()
    anchor_nodes = np.random.choice(num_nodes, NUM_ANCHOR_NODES, replace=False)

    unique_categories = [np.arange(NUM_CATEGORIES)]
    bins = np.r_[np.linspace(-1, NUM_CATEGORIES - 2, num=NUM_CATEGORIES), np.array([np.inf])]

    distance_features_container = []
    for anchor_node in anchor_nodes:
        path_lengths = nx.single_source_shortest_path_length(graph_nx, anchor_node)
        target_nodes, path_length_values = np.array(list(path_lengths.keys())), np.array(list(path_lengths.values()))
        path_length_values = (np.digitize(path_length_values, bins) - 1)

        category_values = np.zeros(shape=(num_nodes, 1))
        category_values[target_nodes, 0] = path_length_values
        
        distance_features = OneHotEncoder(categories=unique_categories, sparse=False).fit_transform(category_values)
        distance_features = np.flip(np.cumsum(np.flip(distance_features, axis=1), axis=1), axis=1)
        distance_features_container.append(distance_features.copy())
    
    distance_features_container = np.concatenate(distance_features_container, axis=1)
    return torch.tensor(distance_features_container, dtype=torch.float)


def prepare_triangle_features(graph):
    graph_nx = nx.Graph(graph.to_networkx())
    triangle_values = np.array(list(nx.cluster.triangles(graph_nx).values()))
    
    unique_categories = [np.arange(NUM_CATEGORIES)]
    bins = np.r_[np.array([0]), np.logspace(0, NUM_CATEGORIES - 2, num=NUM_CATEGORIES - 1, base=2.0), np.array([np.inf])]
    category_values = (np.digitize(triangle_values, bins) - 1).reshape(-1, 1)
    
    triangle_features = OneHotEncoder(categories=unique_categories, sparse=False).fit_transform(category_values)
    triangle_features = np.flip(np.cumsum(np.flip(triangle_features, axis=1), axis=1), axis=1)
    return torch.tensor(triangle_features.copy(), dtype=torch.float)


def prepare_structural_descriptors(graph):
    return torch.cat([
        prepare_degree_features(graph),
        prepare_distance_features(graph),
        prepare_triangle_features(graph),
    ], dim=1)


structural_feature_name_to_prepare_fn = {
    'degree': prepare_degree_features,
    'distance': prepare_distance_features,
    'triangle': prepare_triangle_features,
    'descriptor': prepare_structural_descriptors,
}


def process_homogenous_graph(graph, process_config):
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    if process_config.get('structural_feature_name') is not None:
        structural_feature_name = process_config['structural_feature_name']
        prepare_structural_feature_fn = structural_feature_name_to_prepare_fn[structural_feature_name]
        structural_features = prepare_structural_feature_fn(graph)
        graph.ndata['structural'] = torch.tensor(structural_features, dtype=torch.float)
    
    return graph


dataset_name_to_process_graph_fn = {
    'amazon-computer': process_homogenous_graph,
    'amazon-photo': process_homogenous_graph,
    'coauthor-cs': process_homogenous_graph,
    'coauthor-physics': process_homogenous_graph,
    'cora-ml': process_homogenous_graph,
    'citeseer': process_homogenous_graph,
    'pubmed': process_homogenous_graph,

    'roman-empire': process_homogenous_graph,
    'amazon-ratings': process_homogenous_graph,
    'minesweeper': process_homogenous_graph,
    'workers': process_homogenous_graph,
    'questions': process_homogenous_graph,
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

    'roman-empire': prepare_homogenous_masks,
    'amazon-ratings': prepare_homogenous_masks,
    'minesweeper': prepare_homogenous_masks,
    'workers': prepare_homogenous_masks,
    'questions': prepare_homogenous_masks,
}