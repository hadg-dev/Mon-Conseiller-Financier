import os
from pydantic import field_validator
from pydantic_settings import BaseSettings
from configuration.config import load_config

# Load defaults from config, allow overrides from environment variables

conf = load_config()

class Settings(BaseSettings):
    steps: list[str] | None = None

    @field_validator("steps", mode="before")
    @classmethod
    def parse_steps(cls, value: str | list[str] | None) -> list[str] | None:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    debug: bool = True
    # reporting
    report_training: str = conf.defaults.report_training_to
    mlflow_tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name: str = os.getenv("EXPERIMENT_NAME", "mon_conseiller_financier")
    # Training settings
    nb_epochs: int = conf.defaults.nb_epochs
    model_name: str = conf.defaults.model_name
    learning_rate: float = conf.defaults.learning_rate
    # Dataset settings
    repo_id: str = conf.dataset.repo_id
    training_set: str = conf.dataset.training_set
    validation_set: str = conf.dataset.validation_set
