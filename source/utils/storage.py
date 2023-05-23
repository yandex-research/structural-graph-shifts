import os

class Storage:
    def __init__(self, stage_config):
        self.experiment_root = stage_config.experiment_root
        self.experiment_name = stage_config.experiment_name
        
        self.split_no = stage_config.split_no
        self.init_no = stage_config.init_no

    def retrieve_checkpoint_path(self):
        return f"{self.experiment_root}/{self.experiment_name}/split_{self.split_no}/init_{self.init_no}/best.ckpt"

    def retrieve_checkpoint_root(self):
        return f"{self.experiment_root}/{self.experiment_name}/split_{self.split_no}"

    def construct_checkpoint_path(self, checkpoint_root, init_no):
        return f"{checkpoint_root}/init_{init_no}/best.ckpt"

    def checkpoint_path_exists(self):
        return os.path.exists(self.retrieve_checkpoint_path())

    def remove_checkpoint_path(self):
        if self.checkpoint_path_exists():
            os.remove(self.retrieve_checkpoint_path())