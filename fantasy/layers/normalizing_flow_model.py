import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pyblaze.nn as xnn


flow_name_to_class = {
    'radial': xnn.RadialTransform,
    # 'anything': anything else,
}


class NormalizingFlowModel(xnn.NormalizingFlow):
    def __init__(self, flow_config):
        self.config = flow_config
        
        flow_class = flow_name_to_class[self.config['flow_name']]
        transforms = [
            flow_class(self.config['latent_dim']) 
            for _ in range(self.config['num_layers'])
        ]
        
        super().__init__(transforms)

