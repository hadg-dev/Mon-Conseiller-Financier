import os
import json
import logging
import pandas as pd
from huggingface_hub import snapshot_download, login

DEBUG = True
DATASET_DIR = "../dataset"
DATASET_REPOD_ID = "likhitjuttada/finance-reasoning-sft-dataset"
DATASET_PATTERNS = ["combined_train.jsonl"]

def copy_dataset_locally():
    logging.info(f"Saving dataset in directory: {DATASET_DIR}")

    login()

    snapshot_download(
        repo_id=DATASET_REPOD_ID,
        local_dir=DATASET_DIR,
        repo_type="dataset",
        allow_patterns=DATASET_PATTERNS,
    )

    target_file = next(f for f in os.listdir(DATASET_DIR) if f.endswith((".jsonl")))
    target_file_path = os.path.join(DATASET_DIR, target_file)

    df = pd.read_json(target_file_path, lines=target_file_path.endswith(".jsonl"))
    logging.info(df.head())

    if DEBUG:
        row_data = df.iloc[0].to_dict()
        logging.info(json.dumps(row_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    copy_dataset_locally()


