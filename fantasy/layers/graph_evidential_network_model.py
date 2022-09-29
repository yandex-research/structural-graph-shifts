from matplotlib import projections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pyblaze.nn.functional as X

import dgl
import dgl.nn.pytorch as dglnn

from .general import DeterministicConvolutionSequential, IndependentSequential
from .propagation_model import StandardPropagationModel
from .dirichlet_network_model import DirichletNetworkModel


OBSERVED_LABELS_MASK_NAME = 'train'


class GraphEvidentialNetworkModel(DirichletNetworkModel):
    def __init__(self, num_features, num_classes, model_config):
        super().__init__()
        self.config = model_config
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.impute_config()
        
        self.encoder = IndependentSequential(self.config['encoder_config'])
        self.predictor = IndependentSequential(self.config['predictor_config'])
        self.propagation = StandardPropagationModel(self.config['propagation_config'])

    def impute_config(self):
        self.config['encoder_config']['feature_dims'].insert(0, self.num_features)
        self.config['predictor_config']['feature_dims'].append(self.num_classes)

    def forward(self, graph, features):
        representations = self.encoder(features)
        alphas_feature = torch.exp(torch.clamp(self.predictor(representations), min=-30.0, max=30.0))
        alphas_posterior = 1.0 + self.propagation(graph, alphas_feature)
        return alphas_posterior