from fantasy.utils import general, config

from fantasy.data.dataset import get_dataset_class
from fantasy.data.sampler import get_sampler_class
from fantasy.data.datamodule import get_datamodule_class
from fantasy.modules import get_module_class

from fantasy import experiment


class SeparateExperiment:
    def __init__(self, dataset_config, sampler_config, datamodule_config, method_config, trainer_config, stage_config):
        self.dataset_config = dataset_config
        
        self.sampler_config = sampler_config
        self.datamodule_config = datamodule_config
        
        self.routine_name_to_module_pairs = config.prepare_module_configs(method_config)
        self.trainer_config = trainer_config

        self.stage_config = stage_config

        # fix random state
        general.set_random_seed(self.stage_config.random_seed)

        # after random state is fixed, setup everything
        self.setup_dataset()
        
        self.setup_sampler()
        self.setup_datamodule()

        self.setup_modules()
        self.setup_managers()

    def setup_dataset(self):
        # when the dataset is initialised, it also performs data splits
        dataset_class = get_dataset_class(self.dataset_config.class_name)
        dataset = dataset_class(self.dataset_config.clone())
        self.dataset = dataset

    def setup_sampler(self):
        sampler_class = get_sampler_class(self.sampler_config.class_name)
        self.sampler = None if sampler_class is None else sampler_class(self.sampler_config.clone())

    def setup_datamodule(self):
        datamodule_class = get_datamodule_class(self.datamodule_config.class_name)
        self.datamodule = datamodule_class(self.dataset, self.sampler, self.datamodule_config.clone())

    def setup_modules(self):
        self.routine_name_to_module_instances = {}
        for routine_name in ['train', 'infer']:
            
            self.routine_name_to_module_instances[routine_name] = {}
            for module_name, module_config in self.routine_name_to_module_pairs[routine_name].items():
                module_class = get_module_class(module_config.class_name)
                
                module = module_class(module_config.clone())
                module.setup_storage(self.stage_config.clone())

                self.routine_name_to_module_instances[routine_name][module_name] = module

    def setup_managers(self):
        routine_name_to_manager_class = {'train': experiment.StandardTrainer, 'infer': experiment.StandardTester}
        self.routine_name_to_manager_instances = {}
        
        for routine_name in ['train', 'infer']:    
            self.routine_name_to_manager_instances[routine_name] = {}
            
            for module_name in self.routine_name_to_module_pairs[routine_name].keys():
                manager_class = routine_name_to_manager_class[routine_name]

                manager = manager_class(self.trainer_config.clone())
                self.routine_name_to_manager_instances[routine_name][module_name] = manager


    def run_train_routine(self, manager, module):
        if module.storage.checkpoint_path_exists():                         # here I want to use the fact that every module has the only torch.nn property — model,
            module.load_from_storage()                                      # this should make it possible to read the module state dict regardless the particular module class
            module.storage.remove_checkpoint_path()                         # alternatively, I can transfer this torch.nn property between modules using explicit assignment
        
        print(module.model)
        manager.fit(module, datamodule=self.datamodule)
    
    def run_infer_routine(self, manager, module):
        if module.storage.checkpoint_path_exists():
            module.load_from_storage()
        
        results = manager.test(module, datamodule=self.datamodule)

        merged = {}
        for result in results:
            merged.update(result)
        
        return merged


    def run_train_stage(self):
        # run train routine on train stage
        routine_name = 'train'
        for module_name in self.routine_name_to_module_pairs[routine_name].keys():
            print(module_name)
            manager = self.routine_name_to_manager_instances[routine_name][module_name]
            module = self.routine_name_to_module_instances[routine_name][module_name]
            self.run_train_routine(manager, module)
        
        # run infer routine on train stage
        routine_name = 'infer'
        module_name = 'eval_module'

        manager = self.routine_name_to_manager_instances[routine_name][module_name]
        module = self.routine_name_to_module_instances[routine_name][module_name]
        return self.run_infer_routine(manager, module)

    def run_infer_stage(self):
        # run infer routine on infer stage
        routine_name = 'infer'
        module_name = 'infer_module'

        manager = self.routine_name_to_manager_instances[routine_name][module_name]
        module = self.routine_name_to_module_instances[routine_name][module_name]
        return self.run_infer_routine(manager, module)


    def run(self):
        # the idea behind the experimental pipeline is that 
        # the routine performed at this moment + the module used for this routine 
        # can unambiguously descirbe the current stage
        metrics = self.run_train_stage() if self.stage_config.stage_name == 'train' else self.run_infer_stage()

        if self.stage_config.save_separate_results:
            path = f"{self.trainer_config.default_dir}"
            general.save_separate_results(metrics, path)

        return metrics
