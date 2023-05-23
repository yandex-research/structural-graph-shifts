import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

import dgl
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

from source.modules.base_module import BaseLightningModule
from source.layers import get_model_class
from source.layers.loss import DirichletBayesianLoss, NormalizingFlowLoss

UNCERTAINTY_ESTIMATE_NAMES = ['data', 'knowledge', 'total']
OBSERVED_LABELS_MASK_NAME = 'train'


class StructuralPosteriorNetworkLightningModule(BaseLightningModule):
    def __init__(self, module_config):
        super().__init__(module_config)
        self.config = module_config
        
        self.setup_model()
        self.setup_metrics()
        self.setup_loss()

    def setup_model(self):
        model_class = get_model_class('structural_posterior_network')
        self.model = self.model = model_class(
            self.config.num_node_features, 
            self.config.num_structural_features, 
            self.config.num_classes, 
            self.config.model_config
        )

    def setup_loss(self):
        self.loss = DirichletBayesianLoss(self.config.loss_config)

    def forward(self, batch):
        graph, node_features, structural_features, labels, masks, step_name = batch
        return self.model.forward(graph, node_features, structural_features, labels, masks)

    def get_data_uncertainty(self, batch):
        graph, node_features, structural_features, labels, masks, step_name = batch
        return self.model.get_data_uncertainty(graph, node_features, structural_features, labels, masks)

    def get_knowledge_uncertainty(self, batch):
        graph, node_features, structural_features, labels, masks, step_name = batch
        return self.model.get_knowledge_uncertainty(graph, node_features, structural_features, labels, masks)

    def get_total_uncertainty(self, batch):
        graph, node_features, structural_features, labels, masks, step_name = batch
        return self.model.get_total_uncertainty(graph, node_features, structural_features, labels, masks)

    def general_step(self, batch):        
        outputs = self.forward(batch)
        predictions = self.model.get_predictions(outputs).detach()
        
        estimates = {}
        with torch.no_grad():
            for uncertainty_name in UNCERTAINTY_ESTIMATE_NAMES:
                estimates[uncertainty_name] = self.get_uncertainty(batch, uncertainty_name)
        
        return outputs, predictions, estimates

    def training_step(self, batch, batch_idx):
        outputs, predictions, estimates = self.general_step(batch)

        values = self.track_loss(batch, outputs)
        self.track_classification_performance(batch, predictions)
        
        return values[-1]       # anyway, there is only one value in the list, as we use one dataloader for train step      

    def validation_step(self, batch, batch_idx):
        outputs, predictions, estimates = self.general_step(batch)

        self.track_loss(batch, outputs)
        self.track_classification_performance(batch, predictions)

    def test_step(self, batch, batch_idx):
        outputs, predictions, estimates = self.general_step(batch)

        self.track_classification_performance(batch, predictions)
        self.track_misclassification_detection_performance(batch, estimates, predictions)
        self.track_ood_detection_performance(batch, estimates)
        self.track_aggregated_performance(batch, estimates, predictions)


class StructuralPosteriorFlowLightningModule(BaseLightningModule):
    def __init__(self, module_config):
        super().__init__(module_config)
        self.config = module_config
        
        self.setup_model()
        self.setup_metrics()
        self.setup_loss()

    def setup_model(self):
        model_class = get_model_class('structural_posterior_network')
        self.model = self.model = model_class(
            self.config.num_node_features, 
            self.config.num_structural_features, 
            self.config.num_classes, 
            self.config.model_config
        )

    def setup_loss(self):
        self.loss = DirichletBayesianLoss(self.config.loss_config)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.flows.parameters(), 
            lr=self.config.optimizer_config['lr'], 
        )
        return [optimizer]

    def forward(self, batch):
        graph, node_features, structural_features, labels, masks, step_name = batch
        return self.model.forward(graph, node_features, structural_features, labels, masks)

    def general_step(self, batch):
        outputs = self.forward(batch)
        predictions = outputs.detach()
        
        return outputs, predictions

    def training_step(self, batch, batch_idx):
        outputs, predictions = self.general_step(batch)
        values = self.track_loss(batch, outputs)
        return values[-1]

    def validation_step(self, batch, batch_idx):
        outputs, predictions = self.general_step(batch)
        self.track_loss(batch, outputs)

    def test_step(self, batch, batch_idx):
        pass