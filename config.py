from pathlib import Path
from typing import Union, Text
import yaml
from box import Box

class Config:
    """
    Dot-based access to configuration parameters saved in a YAML file.
    """
    def __init__(self, file: Union[Path, Text], pl_models: dict):
        """ 
        Initialize with MODEL_CLASSES to get the correct model class from the YAML file.
        Get a Box object from the YAML file and populate the current Config object with it. """
        self.pl_models = pl_models
        
        # get a Box object from the YAML file
        with open(str(file), 'r') as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        # manually populate the current Config object with the Box object (since Box inheritance fails)
        for key in cfg.keys():
            setattr(self, key, getattr(cfg, key))

        self.dataset.path = Path(self.dataset.path) # convert to Path object for easier manipulation

        # Correct types in train (ex. lr = 5e-5 is read as string)
        for float_var in ["num_workers", "learning_rate", "weight_decay"]:
            val = getattr(self.train, float_var)
            if type(val) != float:
                setattr(self.train, float_var, float(val))

        # Some attributes could not be defined in config.yml, set them as None
        # self.train.num_workers = getattr(self.train, "num_workers", None)

    def call_model(self, base_model):
        return self.pl_models[base_model]
        
    def to_dict(self):
        return self.cfg.to_dict()