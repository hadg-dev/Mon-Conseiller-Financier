from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from aws.upload_adapter import parse_s3_uri, upload_folder_to_s3


# =====================================================================
# Tests for parse_s3_uri
# =====================================================================

@pytest.mark.parametrize(
    "uri, expected_bucket, expected_prefix",
    [
        ("s3://my-bucket/prefix/path", "my-bucket", "prefix/path"),
        ("s3://my-bucket/", "my-bucket", ""),
        ("s3://my-bucket", "my-bucket", ""),
        ("s3://my-bucket/nested/sub/folder/", "my-bucket", "nested/sub/folder/"),
    ],
)
def test_parse_s3_uri_valid(uri, expected_bucket, expected_prefix):
    bucket, prefix = parse_s3_uri(uri)
    assert bucket == expected_bucket
    assert prefix == expected_prefix


@pytest.mark.parametrize(
    "invalid_uri",
    [
        "https://my-bucket/prefix",
        "s3://",
        "file:///local/path",
        "",
        "not-a-uri",
    ],
)
def test_parse_s3_uri_invalid_raises_value_error(invalid_uri):
    with pytest.raises(ValueError, match="URI S3 invalid"):
        parse_s3_uri(invalid_uri)


# =====================================================================
# Tests for upload_folder_to_s3
# =====================================================================

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.artifact_store_uri = "s3://test-bucket/models/adapter"
    settings.aws_default_region = "us-east-1"
    return settings


def test_upload_folder_missing_artifact_store_uri(mock_settings):
    mock_settings.artifact_store_uri = None

    with pytest.raises(ValueError, match="ARTIFACT_STORE_URI var missing"):
        upload_folder_to_s3(mock_settings)


@patch("aws.upload_adapter.boto3.client")
@patch("aws.upload_adapter.output_root")
def test_upload_folder_missing_adapter_directory(mock_output_root, mock_boto_client, mock_settings, tmp_path):
    mock_output_root.return_value = tmp_path
    # tmp_path / "adapter" does not exist

    with pytest.raises(NotADirectoryError, match="is not valid"):
        upload_folder_to_s3(mock_settings)


@patch("aws.upload_adapter.boto3.client")
@patch("aws.upload_adapter.output_root")
def test_upload_folder_empty_directory_warns_and_skips(mock_output_root, mock_boto_client, mock_settings, tmp_path,
                                                       caplog):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    mock_output_root.return_value = tmp_path

    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    with caplog.at_level("WARNING"):
        upload_folder_to_s3(mock_settings)

    assert f"no files found in {adapter_dir.resolve()}" in caplog.text
    mock_s3.upload_file.assert_not_called()


@patch("aws.upload_adapter.datetime")
@patch("aws.upload_adapter.boto3.client")
@patch("aws.upload_adapter.output_root")
def test_upload_folder_successful_upload_with_prefix(mock_output_root, mock_boto_client, mock_datetime, mock_settings, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    mock_output_root.return_value = tmp_path

    mock_datetime.now.return_value = datetime(2026, 10, 1, 0, 0, 0)

    # Create test files
    file1 = adapter_dir / "config.json"
    file1.write_text("{}")

    nested_dir = adapter_dir / "weights"
    nested_dir.mkdir()
    file2 = nested_dir / "adapter_model.bin"
    file2.write_text("weights")

    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    upload_folder_to_s3(mock_settings, max_workers=2)

    # Verify Boto3 client configuration
    mock_boto_client.assert_called_once()
    assert mock_boto_client.call_args.kwargs["region_name"] == "us-east-1"

    # Verify files uploaded with correct S3 keys
    expected_calls = [
        (str(file1.resolve()), "test-bucket", "models/adapter/20261001_000000/config.json"),
        (str(file2.resolve()), "test-bucket", "models/adapter/20261001_000000/weights/adapter_model.bin"),
    ]

    assert mock_s3.upload_file.call_count == 2
    actual_calls = [call.args for call in mock_s3.upload_file.call_args_list]
    for expected in expected_calls:
        assert expected in actual_calls

@patch("aws.upload_adapter.datetime")
@patch("aws.upload_adapter.boto3.client")
@patch("aws.upload_adapter.output_root")
def test_upload_folder_without_s3_prefix(mock_output_root, mock_boto_client, mock_datetime, mock_settings, tmp_path):
    mock_settings.artifact_store_uri = "s3://test-bucket"

    mock_datetime.now.return_value = datetime(2026, 10, 1, 0, 0, 0)

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    mock_output_root.return_value = tmp_path

    file1 = adapter_dir / "file.txt"
    file1.write_text("hello")

    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    upload_folder_to_s3(mock_settings)

    mock_s3.upload_file.assert_called_once_with(
        str(file1.resolve()),
        "test-bucket",
        "20261001_000000/file.txt",
    )