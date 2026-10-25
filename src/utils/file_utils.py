from pathlib import Path
import logging

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def python_root() -> Path:
    return Path(__file__).resolve().parent

def configuration_root() -> Path:
    root = project_root()
    config_dir = Path(root, "configs")
    return config_dir

def dataset_root(create_if_needed: bool = True) -> Path:
    root = project_root()
    config_dir = Path(root, "dataset")
    if create_if_needed and not config_dir.exists():
        config_dir.mkdir(parents=True)
        logging.info(f"Directory created : {config_dir}")
    return config_dir

def output_root(create_if_needed: bool = True) -> Path:
    root = project_root()
    output_dir = Path(root, "output")
    if create_if_needed and not output_dir.exists():
        output_dir.mkdir(parents=True)
        logging.info(f"Directory created : {output_dir}")
    return output_dir
