from matplotlib import projections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pyblaze.nn.functional as X

import dgl
import dgl.nn.pytorch as dglnn

from .general import ConvolutionSequential, FeedForwardSequential
from .normalizing_flow_model import NormalizingFlowModel
from .dirichlet_network_model import DirichletNetworkModel


OBSERVED_LABELS_MASK_NAME = 'train'


class EvidentialNetworkModel(DirichletNetworkModel):
    def __init__(self, num_features, num_classes, model_config):
        super().__init__()
        self.config = model_config
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.impute_config()
        
        self.preprocessing = FeedForwardSequential(self.config['preprocessing_config'])
        self.encoder = ConvolutionSequential(self.config['encoder_config'])
        self.predictor = FeedForwardSequential(self.config['predictor_config'])

    def impute_config(self):
        self.config['preprocessing_config']['feature_dims'][0] = self.num_features
        self.config['predictor_config']['feature_dims'][-1] = self.num_classes

    def forward(self, graph, features):
        representations = self.encoder(graph, self.preprocessing(features))
        alphas_posterior = 1.0 + torch.exp(torch.clamp(self.predictor(representations), min=-30.0, max=30.0))
        return alphas_posterior