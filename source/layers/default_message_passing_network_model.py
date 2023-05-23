import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import dgl
import dgl.nn.pytorch as dglnn

from .general import ConvolutionSequential, FeedForwardSequential


class DefaultMessagePassingNetworkModel(nn.Module):
    def __init__(self, num_features, num_classes, model_config):
        super().__init__()
        self.config = model_config
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.impute_config()
        
        self.preprocessing = (
            FeedForwardSequential(self.config['preprocessing_config'])
            if self.config['preprocessing_config'] is not None else
            nn.Identity()
        )
        self.encoder = ConvolutionSequential(self.config['encoder_config'])
        self.classifier = FeedForwardSequential(self.config['classifier_config'])

    def impute_config(self):
        if self.config['preprocessing_config'] is not None:
            self.config['preprocessing_config']['feature_dims'][0] = self.num_features
        else:
            self.config['encoder_config']['feature_dims'][0] = self.num_features
        self.config['classifier_config']['feature_dims'][-1] = self.num_classes

    def forward(self, graph, features):
        return self.classifier(self.encoder(graph, self.preprocessing(features)))

    def get_total_uncertainty(self, *args, **kwargs):
        logits = self.forward(*args, **kwargs)
        return D.Categorical(logits=logits).entropy()

    def get_data_uncertainty(self, *args, **kwargs):
        return self.get_total_uncertainty(*args, **kwargs)

    def get_knowledge_uncertainty(self, *args, **kwargs):
        return self.get_total_uncertainty(*args, **kwargs)