import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

import dgl
import dgl.nn.pytorch as dglnn


layer_name_to_class = {
    'gcn': dglnn.GraphConv,
    'gat': dglnn.GATConv,
    'sgc': dglnn.SGConv,
    'sage': dglnn.SAGEConv,
    
    'linear': nn.Linear,
    # 'anything': anything else,
}

activation_name_to_class = {
    'none': nn.Identity,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leaky': nn.LeakyReLU,
}

normalisation_name_to_class = {
    'batch': nn.BatchNorm1d,
    'layer': nn.LayerNorm,
    'none': nn.Identity,
}


def get_indexed_layer_config(idx, feature_dims, config, final):
    return {
        'in_features': feature_dims[idx], 
        'out_features': feature_dims[idx + 1],
        
        'layer_name': config['layer_name'], 
        'layer_args': config['layer_args'],
        
        'activation_name': config['activation_name'], 
        'activation_args': config['activation_args'],

        'normalisation_name': config['normalisation_name'],        
        'residual': config['residual'],
        'dropout': config['dropout'],
        'final': (idx + 1 == len(feature_dims) - 1) and final   # this value is True only if the current layer is the last in the constructed sequential
    }                                                           # and this sequential is treated as final (it predicts logits, projects into latent space, etc.)


class GeneralisedLayer(nn.Module):
    def __init__(self, layer_config):
        super().__init__()
        self.config = layer_config

        layer_class = layer_name_to_class[self.config['layer_name']]
        layer_args = self.config['layer_args']
        self.layer = layer_class(self.config['in_features'], self.config['out_features'], **layer_args)

        activation_class = activation_name_to_class[self.config['activation_name']]
        activation_args = self.config['activation_args']
        self.activation = activation_class(**activation_args) if not self.config['final'] else nn.Identity()
        
        normalisation_class = normalisation_name_to_class[self.config['normalisation_name']]
        self.normalisation = normalisation_class(self.config['out_features'])
        
        self.residual = self.config['residual']
        self.dropout = nn.Dropout(self.config['dropout'])

    def forward(self, graph, features):
        raise NotImplementedError()


class ConvolutionLayer(GeneralisedLayer):
    def __init__(self, layer_config):
        super().__init__(layer_config)

    def forward(self, graph, features):
        # output = self.dropout(self.activation(self.layer(graph, self.normalisation(features))))
        output = self.activation(self.layer(graph, features))
        if self.residual:
            output += features
        return self.dropout(self.normalisation(output))


class FeedForwardLayer(GeneralisedLayer):
    def __init__(self, layer_config):
        super().__init__(layer_config)

    def forward(self, features):
        # output = self.dropout(self.activation(self.layer(self.normalisation(features))))
        output = self.activation(self.layer(features))
        if self.residual:
            output += features
        return self.dropout(self.normalisation(output))


layer_type_to_class = {
    'conv': ConvolutionLayer,
    'ff': FeedForwardLayer,
}


class GeneralisedSequential(nn.Module):
    def __init__(self, sequential_config):
        super().__init__()
        self.config = sequential_config
        self.layers = nn.Sequential()

        for idx in range(len(self.config['feature_dims']) - 1):
            indexed_layer_config = get_indexed_layer_config(idx, self.config['feature_dims'], self.config['layer_config'], self.config['final'])
            layer_class = layer_type_to_class[self.config['type']]
            self.layers.append(layer_class(indexed_layer_config))
    
    def forward(self, graph, features):
        raise NotImplementedError()


class ConvolutionSequential(GeneralisedSequential):
    def __init__(self, sequential_config):
        super().__init__(sequential_config)
        self.drop_edge = DropEdge(self.config.get('dropedge', 0.0))
    
    def forward(self, graph, features):
        graph_updated = self.drop_edge(graph)
        for layer in self.layers:
            features = layer(graph_updated, features)
        return features


class FeedForwardSequential(GeneralisedSequential):
    def __init__(self, sequential_config):
        super().__init__(sequential_config)
    
    def forward(self, features):
        return self.layers(features)
    

class DropEdge():
    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, graph):
        if self.ratio == 0.0:
            return graph
        
        num_edges = graph.num_edges()
        num_edges_drop = int(num_edges * self.ratio)
        
        graph_updated = copy.deepcopy(graph)
        edge_ids = np.random.choice(num_edges, num_edges_drop, replace=False)
        graph_updated.remove_edges(edge_ids)
        return graph_updated


class ReconnectEdge():
    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, graph):
        if self.ratio == 0.0:
            return graph
        
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        num_edges_drop = int(num_edges * self.ratio)
        
        graph_updated = copy.deepcopy(graph)
        edge_ids = np.random.choice(num_edges, num_edges_drop, replace=False)
        graph_updated.remove_edges(edge_ids)

        src = np.random.choice(num_nodes, num_edges_drop)
        dst = np.random.choice(num_nodes, num_edges_drop)
        graph_updated.add_edges(src, dst)
        return dgl.remove_self_loop(graph_updated)
