import torch
from torch.utils import data

import pytorch_lightning as pl
import dgl

from functools import partial


DETERMINISTIC_BATCH_SIZE = None                         # batch size argument is set to None in order to avoid default batching: https://pytorch.org/docs/stable/data.html#disable-automatic-batching


class StochasticLightningDataModule(pl.LightningDataModule):
    def __init__(self, dataset, sampler, datamodule_config):
        super().__init__()

        self.dataset = dataset
        self.sampler = sampler

        self.config = datamodule_config
        self.indices = {name: torch.nonzero(mask).squeeze() for name, mask in self.dataset.masks.items()}

    def setup(self, stage):
        device = self.config.device                                         # the device is `gpu:<index>`
        self.dataset.graph = self.dataset.graph.to(device)

        for name in self.indices.keys():
            self.indices[name] = self.indices[name].to(device)

    def teardown(self, stage):
        device = torch.device('cpu')
        self.dataset.graph = self.dataset.graph.to(device)

        for name in self.indices.keys():
            self.indices[name] = self.indices[name].to(device)

    def get_dataloader(self, name):
        return dgl.dataloading.DataLoader(
            self.dataset.graph,
            self.indices[name], 
            self.sampler, 
            device=self.config.device, 
            shuffle=(name == 'train'), 
            batch_size=self.config.batch_size, 
            # use_uva=True,                             # so far, it sometimes doesn't work and may lead to CUDA errors: https://github.com/dmlc/dgl/issues/3800, https://github.com/dmlc/dgl/issues/3917
            num_workers=self.config.num_workers
        )

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('valid')

    def test_dataloader(self):
        return self.get_dataloader('test')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        input_nodes, output_nodes, blocks = batch
        blocks = [block.to(device) for block in blocks]

    def on_after_batch_transfer(self, batch, dataloader_idx):
        input_nodes, output_nodes, blocks = batch

        features = blocks[0].srcdata['feature']
        labels = blocks[-1].dstdata['label']

        return blocks, features, labels


def graph_collate_fn(dataset, step_name):
    graph = dataset.graph
    masks = dataset.masks

    features = graph.ndata['features']
    labels = graph.ndata['labels']
    
    return graph, features, labels, masks, step_name


class DefaultLightningDataModule(pl.LightningDataModule):
    def __init__(self, dataset, sampler, datamodule_config):
        super().__init__()

        self.dataset = dataset
        self.sampler = sampler

        self.config = datamodule_config

    def setup(self, stage):
        device = self.config.device                                             # device is `gpu:<index>`
        self.dataset.graph = self.dataset.graph.to(device)

    def teardown(self, stage):
        device = torch.device('cpu')
        self.dataset.graph = self.dataset.graph.to(device)

    def get_dataloader(self, step_name):
        dataloader = data.DataLoader(
            [self.dataset],                                                     # here batch size is technically 1
            batch_size=DETERMINISTIC_BATCH_SIZE,
            collate_fn=partial(graph_collate_fn, step_name=step_name)           # for convenience, I also return the step name
        )
        
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('valid')

    def test_dataloader(self):
        return self.get_dataloader('test')


datamodule_name_to_class = {
    'stochastic': StochasticLightningDataModule,
    'default': DefaultLightningDataModule,
    '': None,
}

def get_datamodule_class(datamodule_class_name):
    return datamodule_name_to_class[datamodule_class_name]


# def get_dataloader(self, step_name):
#     # lmao just check this — https://github.com/Lightning-AI/lightning/issues/10809
#     dataloaders = []
#     for mask_name in self.dataset.masks.keys():

#         if step_name in mask_name:
#             collate_fn_per_mask = lambda dataset, mask_name: (
#                 dataset.graph, dataset.masks, mask_name
#             )

#             dataloader = data.DataLoader(
#                 [self.dataset],                         # here batch size is technically 1
#                 batch_size=DETERMINISTIC_BATCH_SIZE,
#                 collate_fn=partial(collate_fn_per_mask, mask_name=mask_name)
#             )
#             dataloaders.append(dataloader)
    
#     return dataloaders if len(dataloaders) != 1 else dataloaders[0]