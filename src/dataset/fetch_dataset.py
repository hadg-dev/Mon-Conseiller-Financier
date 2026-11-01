import os
import json
import logging
import hashlib
import subprocess
import mlflow
import pandas as pd
from huggingface_hub import snapshot_download, login
from settings import Settings
from utils.file_utils import dataset_root

def fetch_dataset_from_hf(settings: Settings) -> None:
    _dataset_dir = str(dataset_root(create_if_needed=True))
    logging.info(f"Saving dataset in directory: {_dataset_dir}")

    login()

    snapshot_download(
        repo_id=settings.repo_id,
        local_dir=_dataset_dir,
        repo_type="dataset",
        allow_patterns= [settings.training_set, settings.validation_set],
    )

    target_file_path = os.path.join(_dataset_dir, settings.training_set)

    logging.info(f"Check head of {settings.training_set}")
    df = pd.read_json(target_file_path, lines=True)
    logging.info(df.head())

    if settings.debug:
        row_data = df.iloc[0].to_dict()
        logging.info(json.dumps(row_data, indent=2, ensure_ascii=False))

def fetch_dataset_version(settings: Settings) -> None:
    """
    Download a version of the training set with DVC, from its data-v* git tag.
    """
    if not settings.data_version:
        raise ValueError("DATA_VERSION var missing.")
    if not settings.data_repo:
        raise ValueError("DATA_REPO var missing.")

    target_file_path = os.path.join(str(dataset_root(create_if_needed=True)), settings.training_set)
    logging.info(f"Downloading dataset version {settings.data_version} to {target_file_path}")

    subprocess.run(
        [
            "dvc", "get",
            f"https://github.com/{settings.data_repo}",
            f"dataset/{settings.training_set}",
            "--rev", settings.data_version,
            "--out", target_file_path,
            "--force",
        ],
        check=True,
    )

    with open(target_file_path, "rb") as f:
        # Same hash as in the dvc pointer file
        data_md5 = hashlib.md5(f.read()).hexdigest()
    data_rows = len(pd.read_json(target_file_path, lines=True))
    logging.info(f"Dataset version {settings.data_version} : {data_rows} rows, md5 {data_md5}")

    if settings.report_training == "mlflow":
        mlflow.log_params({
            "data_version": settings.data_version,
            "data_md5": data_md5,
            "data_rows": data_rows,
        })
