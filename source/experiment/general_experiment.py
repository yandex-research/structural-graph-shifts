from source import experiment
from source.utils import general, config


class GeneralExperiment:
    def __init__(self, dataset_config, datamodule_config, method_config, trainer_config, stage_config):
        self.dataset_config = dataset_config
        self.datamodule_config = datamodule_config
        
        self.method_config = method_config
        self.trainer_config = trainer_config

        self.stage_config = stage_config

        # prepare split and init nos
        assert self.dataset_config.split_no or self.stage_config.num_splits
        self.split_nos = (
            [self.dataset_config.split_no] if not self.stage_config.num_splits 
            else range(1, self.stage_config.num_splits + 1)
        )

        assert self.method_config.init_no or self.stage_config.num_inits
        self.init_nos = (
            [self.method_config.init_no] if not self.stage_config.num_inits 
            else range(1, self.stage_config.num_inits + 1)
        )

    def run(self):
        history = []

        for split_no in self.split_nos:
            for init_no in self.init_nos:
                separate_config = config.sync_configs_during_experiment(
                    self.dataset_config.clone(), 
                    self.datamodule_config.clone(), 
                    
                    self.method_config.clone(), 
                    self.trainer_config.clone(), 
                    
                    self.stage_config.clone(), 
                    
                    split_no, 
                    init_no
                )

                ex = experiment.SeparateExperiment(*separate_config)
                results = ex.run()
                history.append(results)

        if self.stage_config.save_history_results:
            path = f"{self.stage_config.experiment_root}/{self.stage_config.experiment_name}"
            general.save_history_results(history, path)
        
        return history