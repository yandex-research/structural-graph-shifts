import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pyblaze.nn.functional as X

import dgl
import dgl.nn.pytorch as dglnn

from .general import IndependentSequential
from .normalizing_flow_model import NormalizingFlowModel
from .propagation_model import StandardPropagationModel
from .dirichlet_network_model import DirichletNetworkModel


OBSERVED_LABELS_MASK_NAME = 'train'


class GraphPosteriorNetworkModel(DirichletNetworkModel):
    def __init__(self, num_features, num_classes, model_config):
        super().__init__()
        self.config = model_config
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.impute_config()
        
        self.encoder = IndependentSequential(self.config['encoder_config'])
        self.projector = IndependentSequential(self.config['projector_config'])
        
        self.flows = nn.ModuleList([
            NormalizingFlowModel(self.config['flow_config']) 
            for _ in range(self.num_classes)
        ])
        self.propagation = StandardPropagationModel(self.config['propagation_config'])

    def impute_config(self):
        self.config['encoder_config']['feature_dims'].insert(0, self.num_features)

    def get_class_probas(self, observed):
        class_counts = torch.zeros(self.num_classes, device=observed.device)

        for class_idx in range(self.num_classes):
            class_counts[class_idx] = torch.sum((observed == class_idx).type(torch.long))

        class_probas = class_counts / len(observed)
        return class_probas.reshape(1, -1)

    def get_log_density_estimates(self, projections):
        log_density_estimates = torch.zeros(len(projections), self.num_classes, device=projections.device)

        for class_idx in range(self.num_classes):
            projections_transformed, log_det = self.flows[class_idx](projections)
            log_density_estimates[:, class_idx] = X.log_prob_standard_normal(projections_transformed) + log_det

        return log_density_estimates

    def get_evidence_scale(self):
        latent_dim = self.config['flow_config']['latent_dim']
        log_scale = 0.5 * latent_dim * torch.log(4 * torch.tensor(torch.pi))
        return log_scale

    def forward(self, graph, features, labels, masks):
        mask = masks[OBSERVED_LABELS_MASK_NAME]                     # the observed class probas are always derived from the training data
        labels_observed = labels[mask]

        representations = self.encoder(features)
        projections = self.projector(representations)

        log_conditional_density_estimates = self.get_log_density_estimates(projections)
        log_class_probas = torch.log(self.get_class_probas(labels_observed))
        log_density_estimates = log_conditional_density_estimates + log_class_probas
        
        log_scale = self.get_evidence_scale()
        alphas_feature = torch.exp(torch.clamp(log_density_estimates + log_scale, min=-30.0, max=30.0))

        alphas_posterior = 1.0 + self.propagation(graph, alphas_feature)
        return alphas_posterior