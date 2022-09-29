import math

import torch
import torch.nn  as nn
import torch.nn.functional as F

import dgl.nn.pytorch as dglnn
import dgl.function as fn

class StandardPropagationModel(dglnn.APPNPConv):
    def __init__(self, propagation_config):
        self.config = propagation_config
        super().__init__(self.config['num_iters'], self.config['alpha'])


class StochasticMatrix(nn.Module):
    def __init__(self, in_features, out_features, side='right'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.side = 1 if side == 'right' else 0
        
        self.reset_parameters()

    def forward(self, x):
        return x @ F.softmax(self.weight, dim=self.side)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class CompatiblePropagationModel(nn.Module):
    def __init__(self, propagation_config):
        super().__init__()
        self.config = propagation_config

        self.num_iters = self.config['num_iters']
        self.alpha = self.config['alpha']
        self.transform = StochasticMatrix(self.config['num_classes'], self.config['num_classes'], side='right')

    def forward(self, graph, estimates):
        with graph.local_scope():
            degrees = torch.clamp(graph.in_degrees().float(), min=1)
            norm = torch.pow(degrees, -1).reshape(-1, 1).to(estimates.device)
            estimates_init = estimates

            for _ in range(self.num_iters):
                # estimates = self.transform(estimates) * norm
                estimates = self.transform(estimates)
                graph.ndata['estimates'] = estimates
                
                graph.update_all(fn.copy_u('estimates', 'messages'), fn.sum('messages', 'aggregations'))
                aggregations = graph.ndata.pop('aggregations') * norm

                estimates = (1 - self.alpha) * aggregations + self.alpha * estimates_init
            
            return estimates