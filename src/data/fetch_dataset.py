import os
import json
import logging
import pandas as pd
from huggingface_hub import snapshot_download, login
from utils.utils import dataset_root
from configuration.config import load_config

DEBUG = True

def copy_dataset_locally():
    _dataset_dir = str(dataset_root(create_if_needed=True))
    logging.info(f"Saving dataset in directory: {_dataset_dir}")

    login()

    _dataset_config = load_config().dataset

    snapshot_download(
        repo_id=_dataset_config.repo_id,
        local_dir=_dataset_dir,
        repo_type="dataset",
        allow_patterns=_dataset_config.patterns,
    )

    target_file = next(f for f in os.listdir(_dataset_dir) if f.endswith((".jsonl")))
    target_file_path = os.path.join(_dataset_dir, target_file)

    df = pd.read_json(target_file_path, lines=target_file_path.endswith(".jsonl"))
    logging.info(df.head())

    if DEBUG:
        row_data = df.iloc[0].to_dict()
        logging.info(json.dumps(row_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    copy_dataset_locally()


