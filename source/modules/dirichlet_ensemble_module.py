import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

import dgl
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

from source.modules.base_module import BaseLightningModule
from source.modules import module_name_to_class

UNCERTAINTY_ESTIMATE_NAMES = ['data', 'knowledge', 'total']


class DirichletEnsembleLightningModule(BaseLightningModule):
    def __init__(self, module_config):
        super().__init__(module_config)
        self.config = module_config
        
        self.setup_model()
        self.setup_metrics()
        self.setup_loss()

    def setup_model(self):
        module_class = module_name_to_class[self.config.abstract_config['base_name']]
        models = [module_class(self.config.clone()) for _ in range(
                self.config.abstract_config['start_init_no'], 
                self.config.abstract_config['end_init_no'] + 1
            )
        ]
        self.models = nn.ModuleList(models)

    def setup_loss(self):
        pass

    def get_predictions(self, alphas):
        predictions = D.Dirichlet(alphas).mean
        return predictions

    def get_evidence(self, alphas):
        evidence = torch.sum(alphas, dim=1)
        return evidence

    def get_separate_outputs(self, batch):
        separate_outputs = []
        for module in self.models:
            outputs = module.forward(batch)
            separate_outputs.append(outputs)
        
        return separate_outputs

    def get_aggregated_outputs(self, batch):
        separate_outputs = self.get_separate_outputs(batch)
        aggregated_outputs = torch.sum(torch.stack(separate_outputs, dim=0), dim=0)
        return aggregated_outputs

    def forward(self, batch):
        return self.get_aggregated_outputs(batch)

    def get_total_uncertainty(self, batch):
        alphas_posterior = self.get_aggregated_outputs(batch)
        predictions = self.get_predictions(alphas_posterior)
        return D.Categorical(predictions).entropy()

    def get_data_uncertainty(self, batch):
        alphas_posterior = self.get_aggregated_outputs(batch)
        predictions = self.get_predictions(alphas_posterior)
        evidence = self.get_evidence(alphas_posterior).reshape(-1, 1)
        temp = torch.digamma(1 + evidence) - torch.digamma(1 + alphas_posterior)
        return torch.sum(predictions * temp, dim=1)

    def get_knowledge_uncertainty(self, batch):
        alphas_posterior = self.get_aggregated_outputs(batch)
        evidence = self.get_evidence(alphas_posterior)
        return -evidence

    def general_step(self, batch):
        outputs = self.forward(batch)
        predictions = self.get_predictions(outputs)

        estimates = {}
        with torch.no_grad():
            for uncertainty_name in UNCERTAINTY_ESTIMATE_NAMES:
                estimates[uncertainty_name] = self.get_uncertainty(batch, uncertainty_name)
        
        return outputs, predictions, estimates

    def test_step(self, batch, batch_idx):
        outputs, predictions, estimates = self.general_step(batch)

        self.track_classification_performance(batch, predictions)
        self.track_misclassification_detection_performance(batch, estimates, predictions)
        self.track_ood_detection_performance(batch, estimates)
        self.track_aggregated_performance(batch, estimates, predictions)
    
    def load_from_storage(self):
        checkpoint_root = self.storage.retrieve_checkpoint_root()

        for init_no, module in enumerate(self.models, start=1):
            checkpoint_path = self.storage.construct_checkpoint_path(checkpoint_root, init_no)
            module.load_from_checkpoint(checkpoint_path)
