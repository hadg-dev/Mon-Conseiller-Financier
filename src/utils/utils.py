from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def python_root() -> Path:
    return Path(__file__).resolve().parent

def configuration_root() -> Path:
    root = project_root()
    config_dir = Path(root, "configs")
    return config_dir
