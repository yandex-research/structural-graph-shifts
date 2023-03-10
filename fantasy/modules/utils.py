from .default_message_passing_network_module import DefaultMessagePassingNetworkLightningModule
from .default_multilayer_perceptron_module import DefaultMultilayerPerceptronLightningModule

from .evidential_network_module import EvidentialNetworkLightningModule
from .graph_evidential_network_module import GraphEvidentialNetworkLightningModule

from .posterior_network_module import PosteriorNetworkLightningModule, PosteriorFlowLightningModule
from .natural_posterior_network_module import NaturalPosteriorNetworkLightningModule, NaturalPosteriorFlowLightningModule

from .graph_posterior_network_module import GraphPosteriorNetworkLightningModule, GraphPosteriorFlowLightningModule
from .natural_graph_posterior_network_module import NaturalGraphPosteriorNetworkLightningModule, NaturalGraphPosteriorFlowLightningModule

module_name_to_class = {
    'default_message_passing_network': DefaultMessagePassingNetworkLightningModule,
    'default_multilayer_perceptron': DefaultMultilayerPerceptronLightningModule,
    
    'evidential_network': EvidentialNetworkLightningModule,
    'graph_evidential_network': GraphEvidentialNetworkLightningModule,

    'posterior_network': PosteriorNetworkLightningModule,
    'posterior_flow': PosteriorFlowLightningModule,
    
    'graph_posterior_network': GraphPosteriorNetworkLightningModule,
    'graph_posterior_flow': GraphPosteriorFlowLightningModule,

    'natural_posterior_network': NaturalPosteriorNetworkLightningModule,
    'natural_posterior_flow': NaturalPosteriorFlowLightningModule,
    
    'natural_graph_posterior_network': NaturalGraphPosteriorNetworkLightningModule,
    'natural_graph_posterior_flow': NaturalGraphPosteriorFlowLightningModule,

    # '': None
}