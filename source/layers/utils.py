from .default_message_passing_network_model import DefaultMessagePassingNetworkModel
from .structural_message_passing_network_model import StructuralMessagePassingNetworkModel
from .default_multilayer_perceptron_model import DefaultMultilayerPerceptronModel
from .structural_multilayer_perceptron_model import StructuralMultilayerPerceptronModel

from .natural_posterior_network_model import NaturalPosteriorNetworkModel
from .graph_posterior_network_model import GraphPosteriorNetworkModel
from .structural_posterior_network_model import StructuralPosteriorNetworkModel

model_name_to_class = {
    'default_message_passing_network': DefaultMessagePassingNetworkModel,
    'structural_message_passing_network': StructuralMessagePassingNetworkModel,
    'default_multilayer_perceptron': DefaultMultilayerPerceptronModel,
    'structural_multilayer_perceptron': StructuralMultilayerPerceptronModel,
    
    'graph_posterior_network': GraphPosteriorNetworkModel,
    'structural_posterior_network': StructuralPosteriorNetworkModel,
    'natural_posterior_network': NaturalPosteriorNetworkModel,
    # '': None
}