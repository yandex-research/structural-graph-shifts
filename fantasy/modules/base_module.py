import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

from fantasy.utils import storage
from fantasy.metrics import create_metrics_per_problem_name, create_misclassification_targets, create_ood_targets, prepare_masks_per_step, reduce_masks


DETERMINISTIC_BATCH_SIZE = 1
STANDARD_MASK_NAMES = ['_train', '_valid_all', '_valid_in', '_valid_out', '_test_all', '_test_in', '_test_out']


class AveragingMetricsCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        module_name = pl_module.config.module_name
        mask_name = 'valid_all'

        if not pl_module.trainer.sanity_checking:
            loss_averaged = (
                0.5 * trainer.logged_metrics[f'{module_name}/valid_in/loss'] + 
                0.5 * trainer.logged_metrics[f'{module_name}/valid_out/loss']
            )

            pl_module.log(f'{module_name}/{mask_name}/loss_averaged', loss_averaged, on_step=False, on_epoch=True, prog_bar=True, 
            batch_size=DETERMINISTIC_BATCH_SIZE, add_dataloader_idx=False)


class BaseLightningModule(pl.LightningModule):
    def __init__(self, module_config):
        super().__init__()
        self.config = module_config

    def setup_model(self):
        raise NotImplementedError()

    def setup_loss(self):
        raise NotImplementedError()

    def setup_metrics(self):
        metrics = {name: create_metrics_per_problem_name(self.config.metrics_config) for name in STANDARD_MASK_NAMES}
        self.metrics = nn.ModuleDict(metrics)

    def configure_callbacks(self):
        module_name = self.config.module_name

        # averaging_metrics_callback = AveragingMetricsCallback()
        checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=self.trainer.default_root_dir, 
            filename='best',
            monitor=f'module={module_name}/mask=valid_in/value=loss'
        )

        return [checkpoint_callback]
    
    def configure_optimizers(self): # TODO: add scheduler
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.optimizer_config['lr'], 
            weight_decay=self.config.optimizer_config['weight_decay']
        )
        return [optimizer]


    def setup_storage(self, experiment_config):
        self.storage = storage.Storage(experiment_config)
    
    def load_from_storage(self):
        checkpoint_path = self.storage.retrieve_checkpoint_path()
        self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)['state_dict']
        self.load_state_dict(checkpoint)


    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def get_data_uncertainty(self, *args, **kwargs):
        raise NotImplementedError()

    def get_knowledge_uncertainty(self, *args, **kwargs):
        raise NotImplementedError()

    def get_total_uncertainty(self, *args, **kwargs):
        raise NotImplementedError()

    def get_uncertainty(self, batch, uncertainty_name):
        return getattr(self, f'get_{uncertainty_name}_uncertainty')(batch)


    def log_metrics(self, mask_name, problem_name, predictions, targets):
        module_name = self.config.module_name
        for metric_name, metric in self.metrics[f'_{mask_name}'][problem_name].items():
            metric.update(predictions, targets)
            
            condition = (mask_name == 'valid_in' and problem_name == 'classification_ranking')
            self.log(f"module={module_name}/mask={mask_name}/problem={problem_name}/value={metric_name}", metric, 
            on_step=False, on_epoch=True, prog_bar=condition, batch_size=DETERMINISTIC_BATCH_SIZE)

    def log_loss(self, mask_name, outputs, targets):
        module_name = self.config.module_name
        loss = self.loss(outputs, targets)

        self.log(f'module={module_name}/mask={mask_name}/value=loss', loss, 
        on_step=False, on_epoch=True, prog_bar=True, batch_size=DETERMINISTIC_BATCH_SIZE)

        return loss

    
    def track_loss(self, batch, outputs):
        graph, features, labels, masks, step_name = batch
        masks_prepared = prepare_masks_per_step(masks, step_name)

        values = []
        for mask_name, mask in masks_prepared.items():
            loss = self.log_loss(mask_name, outputs[mask], labels[mask])
            values.append(loss)
        
        return values

    def track_classification_performance(self, batch, predictions):
        graph, features, labels, masks, step_name = batch
        masks_prepared = prepare_masks_per_step(masks, step_name)

        for mask_name, mask in masks_prepared.items():
            for problem_name in ['classification_basic', 'classification_ranking']:
                self.log_metrics(mask_name, problem_name, predictions[mask], labels[mask])

        if step_name == 'train':
            return

        # compute metrics on joint split
        mask_name, mask = f'{step_name}_all', reduce_masks(masks_prepared, torch.logical_or)
        for problem_name in ['classification_basic', 'classification_ranking']:
            self.log_metrics(mask_name, problem_name, predictions[mask], labels[mask])

    def track_misclassification_detection_performance(self, batch, estimates, predictions):
        graph, features, labels, masks, step_name = batch
        masks_prepared = prepare_masks_per_step(masks, step_name)
        targets = create_misclassification_targets(predictions, labels, predictions.device)
        
        for mask_name, mask in masks_prepared.items():
            for uncertainty_name, uncertainties in estimates.items():
                problem_name = f'misclassification_detection_using_{uncertainty_name}'
                self.log_metrics(mask_name, problem_name, uncertainties[mask], targets[mask])

        if step_name == 'train':
            return

        # compute metrics on joint split
        mask_name, mask = f'{step_name}_all', reduce_masks(masks_prepared, torch.logical_or)
        for uncertainty_name, uncertainties in estimates.items():
            problem_name = f'misclassification_detection_using_{uncertainty_name}'
            self.log_metrics(mask_name, problem_name, uncertainties[mask], targets[mask])

    def track_ood_detection_performance(self, batch, estimates):
        graph, features, labels, masks, step_name = batch

        if step_name == 'train':
            return

        masks_prepared = prepare_masks_per_step(masks, step_name)
        targets = create_ood_targets(masks_prepared, labels.device)
        
        mask_name, mask = f'{step_name}_all', reduce_masks(masks_prepared, torch.logical_or)
        for uncertainty_name, uncertainties in estimates.items():
            problem_name = f'ood_detection_using_{uncertainty_name}'
            self.log_metrics(mask_name, problem_name, uncertainties[mask], targets[mask])

    def track_aggregated_performance(self, batch, estimates, predictions):
        graph, features, labels, masks, step_name = batch

        if step_name == 'train':
            return

        masks_prepared = prepare_masks_per_step(masks, step_name)
        targets = create_misclassification_targets(predictions, labels, predictions.device)

        mask_name, mask = f'{step_name}_all', reduce_masks(masks_prepared, torch.logical_or)
        for uncertainty_name, uncertainties in estimates.items():
            problem_name = f'aggregated_performance_using_{uncertainty_name}'
            self.log_metrics(mask_name, problem_name, uncertainties[mask], targets[mask])
