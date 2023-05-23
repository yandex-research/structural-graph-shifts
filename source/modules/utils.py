from .default_message_passing_network_module import DefaultMessagePassingNetworkLightningModule
from .structural_message_passing_network_module import StructuralMessagePassingNetworkLightningModule
from .default_multilayer_perceptron_module import DefaultMultilayerPerceptronLightningModule
from .structural_multilayer_perceptron_module import StructuralMultilayerPerceptronLightningModule

from .natural_posterior_network_module import NaturalPosteriorNetworkLightningModule, NaturalPosteriorFlowLightningModule
from .graph_posterior_network_module import GraphPosteriorNetworkLightningModule, GraphPosteriorFlowLightningModule
from .structural_posterior_network_module import StructuralPosteriorNetworkLightningModule, StructuralPosteriorFlowLightningModule

module_name_to_class = {
    'default_message_passing_network': DefaultMessagePassingNetworkLightningModule,
    'structural_message_passing_network': StructuralMessagePassingNetworkLightningModule,
    'default_multilayer_perceptron': DefaultMultilayerPerceptronLightningModule,
    'structural_multilayer_perceptron': StructuralMultilayerPerceptronLightningModule,
    
    'natural_posterior_network': NaturalPosteriorNetworkLightningModule,
    'natural_posterior_flow': NaturalPosteriorFlowLightningModule,
    'graph_posterior_network': GraphPosteriorNetworkLightningModule,
    'graph_posterior_flow': GraphPosteriorFlowLightningModule,
    'structural_posterior_network': StructuralPosteriorNetworkLightningModule,
    'structural_posterior_flow': StructuralPosteriorFlowLightningModule,
    # '': None
}