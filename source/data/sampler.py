import dgl
from dgl import dataloading

class StandardSampler(dataloading.MultiLayerFullNeighborSampler):
    def __init__(self, sampler_config):
        self.config = sampler_config
        super().__init__(self.config.num_layers)


class CustomSampler(dataloading.NeighborSampler):
    def __init__(self, sampler_config):
        self.config = sampler_config
        super().__init__(self.config.fanouts, replace=self.config.replace)


def get_sampler_class(sampler_class):
    sampler_name_to_class = {
        'standard': StandardSampler,
        'custom': CustomSampler,
        # TODO: other samplers
        'none': None,
    }

    return sampler_name_to_class[sampler_class]