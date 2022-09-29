import argparse
import warnings

from fantasy.utils import general, config
from fantasy import experiment


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_config_path', 
        type=str,
        help='configuration file for experiment'
    )

    options = parser.parse_args()
    run_config = config.prepare_run_config(config.read_config(options.run_config_path))

    dataset_config, sampler_config, datamodule_config, method_config, trainer_config, experiment_config = config.process_run_config(run_config)
    stage_name_to_config = config.prepare_stage_configs(experiment_config)

    if experiment_config.do_train:
        stage_config = stage_name_to_config['train']
        general_config = config.sync_configs_before_experiment(
            dataset_config.clone(), 
            
            sampler_config.clone(), 
            datamodule_config.clone(), 
            
            method_config.clone(), 
            trainer_config.clone(), 

            stage_config.clone()
        )

        ex = experiment.GeneralExperiment(*general_config)
        history = ex.run()

    if experiment_config.do_infer:
        stage_config = stage_name_to_config['infer']
        general_config = config.sync_configs_before_experiment(
            dataset_config.clone(), 
            
            sampler_config.clone(), 
            datamodule_config.clone(), 
            
            method_config.clone(), 
            trainer_config.clone(), 

            stage_config.clone()
        )

        ex = experiment.GeneralExperiment(*general_config)
        history = ex.run()
