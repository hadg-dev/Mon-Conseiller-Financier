import hashlib
import json
import pytest
from unittest.mock import MagicMock, patch

from dataset.fetch_dataset import fetch_dataset_version

ROWS = [{"messages": [{"role": "user", "content": "q"}]}, {"messages": [{"role": "user", "content": "q2"}]}]
CONTENT = "".join(json.dumps(row) + "\n" for row in ROWS).encode()


@pytest.fixture
def settings():
    mock_settings = MagicMock()
    mock_settings.data_version = "data-v1"
    mock_settings.data_repo = "owner/repo"
    mock_settings.training_set = "train.jsonl"
    mock_settings.report_training = "none"
    return mock_settings


@pytest.fixture
def fake_dvc_get(tmp_path):
    """Mocks the dvc command, which writes the dataset in the output path."""
    def _write_output(command, check):
        (tmp_path / "train.jsonl").write_bytes(CONTENT)

    with patch("dataset.fetch_dataset.dataset_root", return_value=tmp_path), \
         patch("dataset.fetch_dataset.subprocess.run", side_effect=_write_output) as mock_run:
        yield mock_run


def test_downloads_the_requested_version(settings, fake_dvc_get, tmp_path):
    fetch_dataset_version(settings)

    fake_dvc_get.assert_called_once_with(
        [
            "dvc", "get",
            "https://github.com/owner/repo",
            "dataset/train.jsonl",
            "--rev", "data-v1",
            "--out", str(tmp_path / "train.jsonl"),
            "--force",
        ],
        check=True,
    )


def test_logs_version_hash_and_rows_to_mlflow(settings, fake_dvc_get):
    settings.report_training = "mlflow"

    with patch("dataset.fetch_dataset.mlflow.log_params") as mock_log_params:
        fetch_dataset_version(settings)

    mock_log_params.assert_called_once_with({
        "data_version": "data-v1",
        "data_md5": hashlib.md5(CONTENT).hexdigest(),
        "data_rows": 2,
    })


def test_does_not_log_to_mlflow_when_not_reporting(settings, fake_dvc_get):
    with patch("dataset.fetch_dataset.mlflow.log_params") as mock_log_params:
        fetch_dataset_version(settings)

    mock_log_params.assert_not_called()


@pytest.mark.parametrize("missing", ["data_version", "data_repo"])
def test_fails_when_a_setting_is_missing(settings, fake_dvc_get, missing):
    setattr(settings, missing, None)

    with pytest.raises(ValueError):
        fetch_dataset_version(settings)

    fake_dvc_get.assert_not_called()
