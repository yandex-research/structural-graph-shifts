import os
from pyrsistent import s

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

import dgl
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

from fantasy.modules.base_module import BaseLightningModule
from fantasy.modules import module_name_to_class

UNCERTAINTY_ESTIMATE_NAMES = ['data', 'knowledge', 'total']


class DefaultEnsembleLightningModule(BaseLightningModule):
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
        self.loss = nn.CrossEntropyLoss()

    def get_separate_predictions(self, batch):
        separate_predictions = []
        for module in self.models:
            logits = module.forward(batch)
            probas = F.softmax(logits, dim=1)
            separate_predictions.append(probas)
        
        return separate_predictions

    def get_averaged_predictions(self, batch):
        separate_predictions = self.get_separate_predictions(batch)
        averaged_predictions = torch.mean(torch.stack(separate_predictions, dim=0), dim=0)
        return averaged_predictions

    def forward(self, batch):
        return self.get_averaged_predictions(batch)

    def get_total_uncertainty(self, batch):
        averaged_predictions = self.get_averaged_predictions(batch)
        total_entropy_value = D.Categorical(probs=averaged_predictions).entropy()
        return total_entropy_value

    def get_data_uncertainty(self, batch):
        separate_predictions = self.get_separate_predictions(batch)
        separate_entropy_values = [D.Categorical(probs=predictions).entropy() for predictions in separate_predictions]

        averaged_entropy_value = torch.mean(torch.stack(separate_entropy_values, dim=0), dim=0)
        return averaged_entropy_value

    def get_knowledge_uncertainty(self, batch):
        total_entropy_value = self.get_total_uncertainty(batch)
        averaged_entropy_value = self.get_data_uncertainty(batch)
        return total_entropy_value - averaged_entropy_value

    def general_step(self, batch):
        outputs = self.forward(batch)
        predictions = outputs.detach()

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
