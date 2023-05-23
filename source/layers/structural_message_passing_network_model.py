import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import dgl
import dgl.nn.pytorch as dglnn

from .general import ConvolutionSequential, FeedForwardSequential


class StructuralMessagePassingNetworkModel(nn.Module):
    def __init__(self, num_node_features, num_structural_features, num_classes, model_config):
        super().__init__()
        self.config = model_config
        
        self.num_node_features = num_node_features
        self.num_structural_features = num_structural_features
        self.num_classes = num_classes
        self.impute_config()
        
        self.node_preprocessing = (
            FeedForwardSequential(self.config['node_preprocessing_config'])
            if self.config['node_preprocessing_config'] is not None else
            nn.Identity()
        )
        self.node_encoder = ConvolutionSequential(self.config['node_encoder_config'])
        
        self.structural_preprocessing = (
            FeedForwardSequential(self.config['structural_preprocessing_config'])
            if self.config['structural_preprocessing_config'] is not None else
            nn.Identity()
        )
        self.structural_encoder = FeedForwardSequential(self.config['structural_encoder_config'])
        self.classifier = FeedForwardSequential(self.config['classifier_config'])

    def impute_config(self):
        if self.config['node_preprocessing_config'] is not None:
            self.config['node_preprocessing_config']['feature_dims'][0] = self.num_node_features
        else:
            self.config['node_encoder_config']['feature_dims'][0] = self.num_node_features
        
        if self.config['structural_preprocessing_config'] is not None:
            self.config['structural_preprocessing_config']['feature_dims'][0] = self.num_structural_features
        else:
            self.config['structural_encoder_config']['feature_dims'][0] = self.num_structural_features
        
        self.config['classifier_config']['feature_dims'][-1] = self.num_classes

    def forward(self, graph, node_features, structural_features):
        node_features_encoded = self.node_encoder(graph, self.node_preprocessing(node_features))
        structural_features_encoded = self.structural_encoder(self.structural_preprocessing(structural_features))
        features_encoded = torch.concat([node_features_encoded, structural_features_encoded], dim=1)
        return self.classifier(features_encoded)

    def get_total_uncertainty(self, *args, **kwargs):
        logits = self.forward(*args, **kwargs)
        return D.Categorical(logits=logits).entropy()

    def get_data_uncertainty(self, *args, **kwargs):
        return self.get_total_uncertainty(*args, **kwargs)

    def get_knowledge_uncertainty(self, *args, **kwargs):
        return self.get_total_uncertainty(*args, **kwargs)