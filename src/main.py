import logging
from settings import Settings
import mlflow
from training.trainer import run_training
from dataset.fetch_dataset import fetch_dataset_from_hf
from validation.judge import score_with_llm
from validation.perplexity import score_perplexity
from aws.upload_adapter import upload_folder_to_s3

def manage_mlflow(_settings: Settings) -> None:
    if _settings.report_training == "mlflow":
        if _settings.mlflow_tracking_uri is None:
            raise ValueError("report training to mlflow but mlflow_tracking_uri missing")
        else:
            mlflow.set_tracking_uri(_settings.mlflow_tracking_uri)
            mlflow.set_experiment(_settings.mlflow_experiment_id)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    settings = Settings()
    logging.info(settings)

    steps_factory = {
        "dataset": lambda: fetch_dataset_from_hf(settings),
        "train": lambda: run_training(settings),
        "score_perplexity": lambda: score_perplexity(settings),
        "score_with_llm": lambda: score_with_llm(settings),
        "copy_s3": lambda: upload_folder_to_s3(settings),
    }

    requested_steps = set(settings.steps) if settings.steps else set(steps_factory.keys())

    invalid = [name for name in requested_steps if name not in steps_factory]
    if invalid:
        raise ValueError(
            f"Unknown step(s): {', '.join(invalid)}. "
            f"Valid options: {', '.join(steps_factory.keys())}"
        )

    # Keep the steps in the order declared in steps_factory
    selected_steps = [step for step in steps_factory if step in requested_steps]
    logging.info(f"selected steps: {selected_steps}")

    # setup MLFlow
    manage_mlflow(settings)

    # execute steps
    for step in selected_steps:
        logging.info(f"Executing step: {step}")
        steps_factory[step]()

    logging.info(f"Script done")
