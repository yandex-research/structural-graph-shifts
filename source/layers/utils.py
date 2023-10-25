from .default_message_passing_network_model import DefaultMessagePassingNetworkModel
from .default_multilayer_perceptron_model import DefaultMultilayerPerceptronModel
from .natural_posterior_network_model import NaturalPosteriorNetworkModel
from .graph_posterior_network_model import GraphPosteriorNetworkModel

model_name_to_class = {
    'default_message_passing_network': DefaultMessagePassingNetworkModel,
    'default_multilayer_perceptron': DefaultMultilayerPerceptronModel,
    'graph_posterior_network': GraphPosteriorNetworkModel,
    'natural_posterior_network': NaturalPosteriorNetworkModel,
    # '': None
}