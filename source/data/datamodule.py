import torch
from torch.utils import data

import pytorch_lightning as pl
import dgl

from functools import partial


DETERMINISTIC_BATCH_SIZE = None
# batch size argument is set to None in order to avoid default batching: 
# https://pytorch.org/docs/stable/data.html#disable-automatic-batching


def default_graph_collate_fn(dataset, step_name):
    graph = dataset.graph
    masks = dataset.masks

    features = graph.ndata['features']
    labels = graph.ndata['labels']
    
    return graph, features, labels, masks, step_name


class DefaultLightningDataModule(pl.LightningDataModule):
    def __init__(self, dataset, datamodule_config):
        super().__init__()
        self.dataset = dataset
        self.config = datamodule_config

    def setup(self, stage):
        device = self.config.device
        self.dataset.graph = self.dataset.graph.to(device)

    def teardown(self, stage):
        device = torch.device('cpu')
        self.dataset.graph = self.dataset.graph.to(device)

    def get_dataloader(self, step_name):
        dataloader = data.DataLoader(
            [self.dataset],
            batch_size=DETERMINISTIC_BATCH_SIZE,
            collate_fn=partial(default_graph_collate_fn, step_name=step_name)
        )
        
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('valid')

    def test_dataloader(self):
        return self.get_dataloader('test')


datamodule_name_to_class = {
    'default': DefaultLightningDataModule,
}

def get_datamodule_class(datamodule_class_name):
    return datamodule_name_to_class[datamodule_class_name]
