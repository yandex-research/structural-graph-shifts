import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pyblaze.nn.functional as X

import dgl
import dgl.nn.pytorch as dglnn

from .general import IndependentSequential
from .normalizing_flow_model import NormalizingFlowModel
from .propagation_model import StandardPropagationModel, CompatiblePropagationModel
from .dirichlet_network_model import DirichletNetworkModel


OBSERVED_LABELS_MASK_NAME = 'train'


class NaturalCompatiblePosteriorNetworkModel(DirichletNetworkModel):
    def __init__(self, num_features, num_classes, model_config):
        super().__init__()
        self.config = model_config
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.impute_config()
        
        self.encoder = IndependentSequential(self.config['encoder_config'])
        self.projector = IndependentSequential(self.config['projector_config'])
        self.predictor = IndependentSequential(self.config['predictor_config'])

        self.flow = NormalizingFlowModel(self.config['flow_config'])
        self.evidence_propagation = StandardPropagationModel(self.config['evidence_propagation_config'])
        self.probas_propagation = CompatiblePropagationModel(self.config['probas_propagation_config'])

    def impute_config(self):
        self.config['encoder_config']['feature_dims'].insert(0, self.num_features)
        self.config['predictor_config']['feature_dims'].append(self.num_classes)
        self.config['probas_propagation_config']['num_classes'] = self.num_classes

    def init_probas_propagation(self, values):
        with torch.no_grad():
            self.probas_propagation.transform.weight.copy_(values)

    def get_log_density_estimates(self, projections):
        log_density_estimates = torch.zeros(len(projections), self.num_classes, device=projections.device)

        projections_transformed, log_det = self.flow(projections)
        log_density_estimates = X.log_prob_standard_normal(projections_transformed) + log_det

        return log_density_estimates.reshape(-1, 1)

    def get_evidence_scale(self):
        latent_dim = self.config['flow_config']['latent_dim']
        log_scale = 0.5 * latent_dim * torch.log(4 * torch.tensor(torch.pi))
        return log_scale

    def forward(self, graph, features):
        representations = self.encoder(features)
        projections = self.projector(representations)
        
        log_density_estimates = self.get_log_density_estimates(projections)
        probas = F.softmax(self.predictor(representations), dim=1)
        probas_propagated = self.probas_propagation(graph, probas)
        
        log_scale = self.get_evidence_scale()
        evidence =  torch.exp(torch.clamp(log_density_estimates + log_scale, min=-30.0, max=30.0))
        evidence_propagated = self.evidence_propagation(graph, evidence)
        
        alphas_propagated = probas_propagated * evidence_propagated
        alphas_posterior = 1.0 + alphas_propagated
        return alphas_posterior