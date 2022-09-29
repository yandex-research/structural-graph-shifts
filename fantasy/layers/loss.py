from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class ExpectedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, alphas, targets, *args, **kwargs):
        alphas_norm = torch.sum(alphas, dim=1)
        alphas_true = torch.gather(alphas, dim=1, index=targets.reshape(-1, 1)).squeeze()   # for each row, take value in `alphas` at index specified by `targets`
        
        uce = torch.digamma(alphas_norm) - torch.digamma(alphas_true)
        return torch.mean(uce)


class DirichletEntropyRegularisation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, alphas, *args, **kwargs):
        reg = D.Dirichlet(alphas).entropy()
        return torch.mean(reg)


class DirichletBayesianLoss(nn.Module):
    def __init__(self, loss_config):
        super().__init__()
        self.config = loss_config
        self.uce = ExpectedCrossEntropyLoss()
        self.reg = DirichletEntropyRegularisation()

    def forward(self, alphas, targets, *args, **kwargs):
        return self.uce(alphas, targets) - self.config['beta'] * self.reg(alphas)


class NormalizingFlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, log_density_estimates, *args, **kwargs):
        return -torch.mean(log_density_estimates)