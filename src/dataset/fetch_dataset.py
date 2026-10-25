import os
import json
import logging
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

