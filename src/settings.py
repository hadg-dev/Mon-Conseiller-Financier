import os
from pydantic import Field, field_validator
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
    mlflow_experiment_id: str = os.getenv("MLFLOW_EXPERIMENT_ID", "mon_conseiller_financier")
    # Training settings
    nb_epochs: int = conf.defaults.nb_epochs
    model_name: str = conf.defaults.model_name
    learning_rate: float = conf.defaults.learning_rate
    # Dataset settings
    repo_id: str = conf.dataset.repo_id
    training_set: str = conf.dataset.training_set
    validation_set: str = conf.dataset.validation_set
    # Dataset version : DVC reads the data-v* tag from this GitHub repo (owner/name)
    data_version: str | None = os.getenv("DATA_VERSION")
    data_repo: str | None = os.getenv("DATA_REPO")
    # AWS S3
    aws_access_key_id: str | None = Field(default=os.getenv("AWS_ACCESS_KEY_ID"), repr=False)
    aws_secret_access_key: str | None = Field(default=os.getenv("AWS_SECRET_ACCESS_KEY"), repr=False)
    artifact_store_uri: str | None = os.getenv("ARTIFACT_STORE_URI")
    aws_default_region: str = os.getenv("AWS_DEFAULT_REGION", "eu-west-3")
