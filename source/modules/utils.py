from .default_message_passing_network_module import DefaultMessagePassingNetworkLightningModule
from .default_multilayer_perceptron_module import DefaultMultilayerPerceptronLightningModule
from .natural_posterior_network_module import NaturalPosteriorNetworkLightningModule, NaturalPosteriorFlowLightningModule
from .graph_posterior_network_module import GraphPosteriorNetworkLightningModule, GraphPosteriorFlowLightningModule

module_name_to_class = {
    'default_message_passing_network': DefaultMessagePassingNetworkLightningModule,
    'default_multilayer_perceptron': DefaultMultilayerPerceptronLightningModule,
    'natural_posterior_network': NaturalPosteriorNetworkLightningModule,
    'natural_posterior_flow': NaturalPosteriorFlowLightningModule,
    'graph_posterior_network': GraphPosteriorNetworkLightningModule,
    'graph_posterior_flow': GraphPosteriorFlowLightningModule,
    # '': None
}