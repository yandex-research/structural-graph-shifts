import pytorch_lightning as pl


class StandardTester(pl.Trainer):
    def __init__(self, tester_config, enable_logging=False):
        self.config = tester_config
        super().__init__(
            logger=enable_logging, 
            enable_checkpointing=False,
            
            gpus=[self.config.gpu_index],
            default_root_dir=self.config.default_dir,
        )
