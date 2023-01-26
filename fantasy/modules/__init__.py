from .utils import module_name_to_class
from .default_ensemble_module import DefaultEnsembleLightningModule
from .monte_carlo_ensemble_module import MonteCarloEnsembleLightningModule
from .dirichlet_ensemble_module import DirichletEnsembleLightningModule
from .dirichlet_mixture_module import DirichletMixtureLightningModule
# from .some_module import SomeModule

module_name_to_class['default_ensemble'] = DefaultEnsembleLightningModule
module_name_to_class['monte_carlo_ensemble'] = MonteCarloEnsembleLightningModule
module_name_to_class['dirichlet_ensemble'] = DirichletEnsembleLightningModule
module_name_to_class['dirichlet_mixture'] = DirichletMixtureLightningModule
# module_name_to_class['some'] = SomeModule

def get_module_class(module_class_name):
    return module_name_to_class[module_class_name]