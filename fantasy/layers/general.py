import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

import dgl
import dgl.nn.pytorch as dglnn


layer_name_to_class = {
    'sage': dglnn.SAGEConv,
    'gat': dglnn.GATConv,
    'sgc': dglnn.SGConv,

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
        self.normalisation = normalisation_class(self.config['in_features'])
        
        self.residual = self.config['residual']
        self.dropout = nn.Dropout(self.config['dropout'])

    def forward(self, graph, features):
        raise NotImplementedError()


class ConvolutionLayer(GeneralisedLayer):
    def __init__(self, layer_config):
        super().__init__(layer_config)

    def forward(self, graph, features):
        output = self.dropout(self.activation(self.layer(graph, self.normalisation(features))))
        return output if not self.residual else output + features


class FeedForwardLayer(GeneralisedLayer):
    def __init__(self, layer_config):
        super().__init__(layer_config)

    def forward(self, features):
        output = self.dropout(self.activation(self.layer(self.normalisation(features))))
        return output if not self.residual else output + features


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
    
    def forward(self, graph, features):
        for layer in self.layers:
            features = layer(graph, features)

        return features


class FeedForwardSequential(GeneralisedSequential):
    def __init__(self, sequential_config):
        super().__init__(sequential_config)
    
    def forward(self, features):
        return self.layers(features)


# class StochasticGeneralisedSequential(GeneralisedSequential):
#     def __init__(self, sequential_config):
#         super().__init__(sequential_config)

#     def forward(self, blocks, features):
#         for block, layer in zip(blocks, self.layers):
#             features = layer(block, features)

#         return features
