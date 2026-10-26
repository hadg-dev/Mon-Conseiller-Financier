import os
import json
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

# Adjust import according to your actual module structure
from dataset.fetch_dataset import fetch_dataset_from_hf


@pytest.fixture
def mock_settings():
    """Returns a mock Settings instance."""
    settings = MagicMock()
    settings.repo_id = "test-org/test-dataset"
    settings.training_set = "train.jsonl"
    settings.validation_set = "val.jsonl"
    settings.debug = False
    return settings


@pytest.fixture
def mock_dependencies():
    """Mocks all external module dependencies inside dataset_fetcher."""
    with patch("dataset.fetch_dataset.dataset_root") as mock_dataset_root, \
         patch("dataset.fetch_dataset.login") as mock_login, \
         patch("dataset.fetch_dataset.snapshot_download") as mock_snapshot, \
         patch("dataset.fetch_dataset.pd.read_json") as mock_read_json, \
         patch("dataset.fetch_dataset.logging.info") as mock_log_info:

        # Default dataset root returns a dummy directory
        mock_dataset_root.return_value = "/dummy/dataset/path"

        # Default DataFrame with dummy records
        sample_df = pd.DataFrame([
            {"id": 1, "text": "sample row one"},
            {"id": 2, "text": "sample row two"}
        ])
        mock_read_json.return_value = sample_df

        yield {
            "dataset_root": mock_dataset_root,
            "login": mock_login,
            "snapshot_download": mock_snapshot,
            "read_json": mock_read_json,
            "log_info": mock_log_info,
            "df": sample_df,
        }


def test_fetch_dataset_from_hf_success(mock_settings, mock_dependencies):
    fetch_dataset_from_hf(mock_settings)

    # 1. Ensure directory resolution ran with create_if_needed
    mock_dependencies["dataset_root"].assert_called_once_with(create_if_needed=True)

    # 2. Ensure Hugging Face login was invoked
    mock_dependencies["login"].assert_called_once()

    # 3. Check snapshot_download arguments
    mock_dependencies["snapshot_download"].assert_called_once_with(
        repo_id="test-org/test-dataset",
        local_dir="/dummy/dataset/path",
        repo_type="dataset",
        allow_patterns=["train.jsonl", "val.jsonl"],
    )

    # 4. Check DataFrame reader call
    expected_path = os.path.join("/dummy/dataset/path", "train.jsonl")
    mock_dependencies["read_json"].assert_called_once_with(expected_path, lines=True)


def test_fetch_dataset_from_hf_debug_enabled(mock_settings, mock_dependencies):
    mock_settings.debug = True

    fetch_dataset_from_hf(mock_settings)

    expected_row = json.dumps(
        {"id": 1, "text": "sample row one"},
        indent=2,
        ensure_ascii=False
    )

    # Extract all string arguments passed to logging.info
    logged_messages = [
        call.args[0]
        for call in mock_dependencies["log_info"].call_args_list
        if isinstance(call.args[0], str)
    ]

    assert expected_row in logged_messages


def test_fetch_dataset_from_hf_empty_dataframe(mock_settings, mock_dependencies):
    mock_settings.debug = True
    # If the file is empty, df.iloc[0] raises IndexError
    mock_dependencies["read_json"].return_value = pd.DataFrame()

    with pytest.raises(IndexError):
        fetch_dataset_from_hf(mock_settings)