from pathlib import Path
import yaml
from utils.utils import configuration_root
from pydantic import BaseModel

CONFIG_FILE_NAME = "training_conf.yaml"

class TrainingDefaultsConfig(BaseModel):
    nb_epochs: int
    learning_rate: float
    model_name: str
    report_training_to: str

class TrainingConfig(BaseModel):
    defaults: TrainingDefaultsConfig

def load_config() -> TrainingDefaultsConfig:
    config_file = Path(configuration_root()).joinpath(CONFIG_FILE_NAME)
    if not config_file.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
        cfg = TrainingConfig(**raw_cfg)
        return cfg.defaults

cfgg = load_config()
print(cfgg)