import pytorch_lightning as pl


class StandardTrainer(pl.Trainer):
    def __init__(self, trainer_config, enable_logging=True):
        self.config = trainer_config
        super().__init__(
            logger=enable_logging, 
            enable_checkpointing=False,
            
            gpus=[self.config.gpu_index], 
            default_root_dir=self.config.default_dir,

            max_epochs=self.config.num_epochs, 
            log_every_n_steps=self.config.log_frequency,
        )
