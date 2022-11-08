from .default_message_passing_network_model import DefaultMessagePassingNetworkModel
from .default_multilayer_perceptron_model import DefaultMultilayerPerceptronModel

from .evidential_network_model import EvidentialNetworkModel
from .graph_evidential_network_model import GraphEvidentialNetworkModel

from .posterior_network_model import PosteriorNetworkModel
from .natural_posterior_network_model import NaturalPosteriorNetworkModel

from .graph_posterior_network_model import GraphPosteriorNetworkModel
from .natural_graph_posterior_network_model import NaturalGraphPosteriorNetworkModel

# from .compatible_posterior_network_model import CompatiblePosteriorNetworkModel
# from .natural_compatible_posterior_network_model import NaturalCompatiblePosteriorNetworkModel

# from .supreme_posterior_network_model import SupremePosteriorNetworkModel
# from .natural_supreme_posterior_network_model import NaturalSupremePosteriorNetworkModel

model_name_to_class = {
    'default_message_passing_network': DefaultMessagePassingNetworkModel,
    'default_multilayer_perceptron': DefaultMultilayerPerceptronModel,
    
    'evidential_network': EvidentialNetworkModel,
    'graph_evidential_network': GraphEvidentialNetworkModel,

    'posterior_network': PosteriorNetworkModel,
    'graph_posterior_network': GraphPosteriorNetworkModel,

    'natural_posterior_network': NaturalPosteriorNetworkModel,
    'natural_graph_posterior_network': NaturalGraphPosteriorNetworkModel,
    
    # 'compatible_posterior_network': CompatiblePosteriorNetworkModel,
    # 'natural_compatible_posterior_network': NaturalCompatiblePosteriorNetworkModel,

    # 'supreme_posterior_network': SupremePosteriorNetworkModel,
    # 'natural_supreme_posterior_network': NaturalSupremePosteriorNetworkModel,
    # '': None
}