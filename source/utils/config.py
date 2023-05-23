import copy
import json
import yaml

from dataclasses import dataclass, asdict


class BaseConfig:
    def clone(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return asdict(self)
    
    # maybe some other functional which has to be shared across config classes


@dataclass
class DatasetConfiguration(BaseConfig):
    class_name: str = 'standard'
    dataset_name: str = None
    
    num_node_features: int = None
    num_structural_features: int = None
    num_classes: int = None

    split_mode: str = 'random'
    split_config: dict = None
    
    random_seed: int = 42
    split_no: int = None
    default_dir: str = './datasets'

    prepare_config: dict = None
    process_config: dict = None


@dataclass
class SamplerConfiguration(BaseConfig):
    class_name: str = 'none'

    # used only when class_name = 'full_neighbor' 
    num_layers: int = None          # sync with module config
    
    # used only when class_name = ...
    fanouts: list = None
    replace: bool = False
    # ...


@dataclass
class DatamoduleConfiguration(BaseConfig):
    class_name: str = 'stochastic'

    batch_size: int = 64
    device: str = None              # sync with trainer config
    num_workers: int = 0


@dataclass
class ModuleConfiguration(BaseConfig):
    num_node_features: int = None           # sync with method config
    num_structural_features: int = None     # sync with method config
    num_classes: int = None                 # sync with method config
    num_convolutions: int = None            # sync with method config

    device: str = None                      # sync with method config
    init_no: int = None                     # sync with method config

    module_name: str = None
    class_name: str = None

    optimizer_config: dict = None
    model_config: dict = None

    metrics_config: dict = None
    loss_config: dict = None
    
    abstract_config: dict = None    # some other configs that are specific for each module (base module class for ensembles, etc.)


@dataclass
class MethodConfiguration(BaseConfig):
    num_node_features: int = None           # sync with dataset config
    num_structural_features: int = None     # sync with dataset config
    num_classes: int = None                 # sync with dataset config
    num_convolutions: int = None

    device: str = None                      # sync with trainer config
    init_no: int = None

    train_routine: dict = None
    infer_routine: dict = None

    # warmup_module_config: dict = None
    # train_module_config: dict = None
    # finetune_module_config: dict = None
    
    # eval_module_config: dict = None
    # infer_module_config: dict = None


@dataclass
class TrainerConfiguration(BaseConfig):
    gpu_index: int = 5
    
    num_epochs: int = 100
    log_frequency: int = 50
    default_dir: str = None         # sync with experiment config


@dataclass
class StageConfiguration(BaseConfig):
    experiment_name: str = None     # sync with super experiment config
    experiment_root: str = None     # sync with super experiment config
    random_seed: int = None         # sync with super experiment config

    stage_name: str = None

    split_no: int = None
    init_no: int = None

    num_inits: int = 5
    num_splits: int = 1
    
    save_separate_results: bool = True
    save_history_results: bool = True


@dataclass
class ExperimentConfiguration(BaseConfig):
    experiment_name: str = None
    experiment_root: str = './experiments'
    random_seed: int = 42

    do_train: bool = True
    train_stage: dict = None

    do_infer: bool = True
    infer_stage: dict = None


def read_config(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def save_config(config, path):
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True, sort_keys=False)


def pprint_config(config):
    print(json.dumps(config, indent=2))


def prepare_run_config(run_config):
    for config_name in ['dataset', 'sampler', 'datamodule', 'method', 'trainer', 'experiment']:
        run_config[config_name] = f"{run_config['root']}/{run_config[config_name]}"
    
    return run_config


def process_run_config(run_config):
    dataset_config = DatasetConfiguration(**read_config(run_config['dataset']))

    sampler_config = SamplerConfiguration(**read_config(run_config['sampler']))
    datamodule_config = DatamoduleConfiguration(**read_config(run_config['datamodule']))
    
    method_config = MethodConfiguration(**read_config(run_config['method']))
    trainer_config = TrainerConfiguration(**read_config(run_config['trainer']))
    
    experiment_config = ExperimentConfiguration(**read_config(run_config['experiment']))
    experiment_config.experiment_name = run_config['name']

    return dataset_config, sampler_config, datamodule_config, method_config, trainer_config, experiment_config


def sync_configs_before_experiment(dataset_config, sampler_config, datamodule_config, method_config, trainer_config, experiment_config):
    sampler_config.num_layers = method_config.num_convolutions
    datamodule_config.device = f"cuda:{trainer_config.gpu_index}"

    method_config.num_node_features = dataset_config.num_node_features
    method_config.num_structural_features = dataset_config.num_structural_features
    method_config.num_classes = dataset_config.num_classes
    method_config.device = f"cuda:{trainer_config.gpu_index}"

    trainer_config.default_dir = f"{experiment_config.experiment_root}/{experiment_config.experiment_name}"
    trainer_config.enable_logging = (experiment_config.experiment_name == 'train')

    return dataset_config, sampler_config, datamodule_config, method_config, trainer_config, experiment_config


def prepare_stage_configs(experiment_config):
    train_stage_config = StageConfiguration(**experiment_config.train_stage)
    infer_stage_config = StageConfiguration(**experiment_config.infer_stage)

    for stage_config in [train_stage_config, infer_stage_config]:
        
        stage_config.experiment_root = experiment_config.experiment_root
        stage_config.experiment_name = experiment_config.experiment_name
        stage_config.random_seed = experiment_config.random_seed
    
    return {'train': train_stage_config, 'infer': infer_stage_config}


def impute_method_config(module_config, method_config):
    module_config.num_node_features = method_config.num_node_features
    module_config.num_structural_features = method_config.num_structural_features
    module_config.num_classes = method_config.num_classes
    module_config.num_convolutions = method_config.num_convolutions
            
    module_config.device = method_config.device
    module_config.init_no = method_config.init_no

    if module_config.metrics_config is not None:
        for problem_name in ['classification_basic', 'classification_ranking']:
            module_config.metrics_config[problem_name]['metric_args']['num_classes'] = module_config.num_classes

    return module_config


def prepare_module_configs(method_config):
    # module configs on train stage can be arbitrary
    train_module_configs = {}

    for module_name, module_config in method_config.train_routine.items():
        # module name can be `warmup_module`, `train_module`, etc.
        module_config = ModuleConfiguration(**module_config)
        module_config = impute_method_config(module_config, method_config)
        
        module_config.module_name = module_name.split('_')[0]
        train_module_configs[module_name] = module_config
    
    # module configs on infer stage also can be arbitrary, but they are set the same as train if contain None
    infer_module_configs = {}

    for module_name, module_config in method_config.infer_routine.items():
        if module_config is not None:
            module_config = ModuleConfiguration(**module_config)
            module_config = impute_method_config(module_config, method_config)
        else:
            module_config = train_module_configs['train_module'].clone()       
        
        module_config.module_name = module_name.split('_')[0]
        infer_module_configs[module_name] = module_config

    return {'train': train_module_configs, 'infer': infer_module_configs}

    
def sync_configs_during_experiment(dataset_config, sampler_config, datamodule_config, method_config, trainer_config, stage_config, split_no, init_no):
    dataset_config.split_no = split_no
    method_config.init_no = init_no

    stage_config.split_no = dataset_config.split_no
    stage_config.init_no = method_config.init_no
    stage_config.random_seed += stage_config.init_no

    trainer_config.default_dir = f"{stage_config.experiment_root}/{stage_config.experiment_name}/split_{stage_config.split_no}/init_{stage_config.init_no}"

    return dataset_config, sampler_config, datamodule_config, method_config, trainer_config, stage_config
