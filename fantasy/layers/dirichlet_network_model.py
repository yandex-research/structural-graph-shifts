from matplotlib import projections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pyblaze.nn.functional as X

import dgl
import dgl.nn.pytorch as dglnn


class DirichletNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_predictions(self, alphas):
        predictions = D.Dirichlet(alphas).mean
        return predictions

    def get_evidence(self, alphas):
        evidence = torch.sum(alphas, dim=1)
        return evidence

    def get_total_uncertainty(self, *args, **kwargs):
        alphas_posterior = self.forward(*args, **kwargs)
        predictions = self.get_predictions(alphas_posterior)
        return D.Categorical(predictions).entropy()

    def get_data_uncertainty(self, *args, **kwargs):
        alphas_posterior = self.forward(*args, **kwargs)
        predictions = self.get_predictions(alphas_posterior)
        evidence = self.get_evidence(alphas_posterior).reshape(-1, 1)
        temp = torch.digamma(1 + evidence) - torch.digamma(1 + alphas_posterior)
        return torch.sum(predictions * temp, dim=1)

    def get_knowledge_uncertainty(self, *args, **kwargs):
        # alphas_posterior = self.forward(*args, **kwargs)
        # evidence = self.get_evidence(alphas_posterior)
        # return -evidence
        total_entropy_value = self.get_total_uncertainty(*args, **kwargs)
        expected_entropy_value = self.get_data_uncertainty(*args, **kwargs)
        return total_entropy_value - expected_entropy_value