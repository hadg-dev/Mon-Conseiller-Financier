import logging
import boto3
from botocore.config import Config
from settings import Settings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from utils.file_utils import output_root
from urllib.parse import urlparse
from datetime import datetime

def parse_s3_uri(s3_uri: str):
    """
    Split S3 URI to (bucket, prefix).
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"URI S3 invalid : {s3_uri}. expected format : s3://mon-bucket/optionnel-prefix")

    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket_name, prefix


def upload_folder_to_s3(settings: Settings, max_workers: int = 8):
    """
    Upload directory to S3.
    """
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_uri = settings.artifact_store_uri
    if not target_uri:
        raise ValueError("ARTIFACT_STORE_URI var missing.")

    bucket_name, base_prefix = parse_s3_uri(target_uri)
    region_name = settings.aws_default_region

    if base_prefix:
        final_prefix = f"{base_prefix}/{timestamp_folder}"
    else:
        final_prefix = timestamp_folder

    # Boto3 client
    s3_client = boto3.client(
        "s3",
        region_name=region_name,
        config=Config(retries={"max_attempts": 5, "mode": "standard"})
    )

    # get output
    adapter_path = Path(output_root(), "adapter").resolve()
    if not adapter_path.is_dir():
        raise NotADirectoryError(f"Pth {adapter_path} is not valid.")

    # list files to upload
    files_to_upload = []
    for file_path in adapter_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(adapter_path).as_posix()
            s3_key = f"{final_prefix}/{relative_path}"
            files_to_upload.append((str(file_path), s3_key))

    if not files_to_upload:
        logging.warning(f"no files found in {adapter_path}.")
        return

    destination_uri = f"s3://{bucket_name}/{final_prefix}"
    logging.info(f"Source   : {adapter_path}")
    logging.info(f"Target   : {destination_uri}")
    logging.info(f"Sending {len(files_to_upload)} file(s) (using {max_workers} threads)...")

    # 4. Sending
    def _upload(item):
        src, key = item
        s3_client.upload_file(src, bucket_name, key)
        return key

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for uploaded_key in executor.map(_upload, files_to_upload):
            logging.info(f"  ✓ {uploaded_key}")

    logging.info("upload completed !")

#if __name__ == "__main__":
#    upload_folder_to_s3()