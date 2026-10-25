import logging
from settings import Settings
import mlflow
from training.trainer import run_training
from dataset.fetch_dataset import fetch_dataset_from_hf
from validation.model_validation import run_validation

def manage_mlflow(_settings: Settings) -> None:
    if _settings.report_training == "mlflow":
        if _settings.mlflow_tracking_uri is None:
            raise ValueError("report training to mlflow but mlflow_tracking_uri missing")
        else:
            mlflow.set_tracking_uri(_settings.mlflow_tracking_uri)
            mlflow.set_experiment(_settings.experiment_name)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    settings = Settings()
    logging.info(settings)

    steps_factory = {
        "dataset": lambda: fetch_dataset_from_hf(settings),
        "train": lambda: run_training(settings),
        "validate": lambda: run_validation(settings),
        #"copy": TODO copy lora to S3
    }

    selected_steps = settings.steps or list(steps_factory.keys())
    invalid = [name for name in selected_steps if name not in steps_factory]
    if invalid:
        raise ValueError(
            f"Unknown step(s): {', '.join(invalid)}. "
            f"Valid options: {', '.join(steps_factory.keys())}"
        )

    # setup MLFlow
    manage_mlflow(settings)

    # execute steps
    for step in selected_steps:
        steps_factory[step]()
