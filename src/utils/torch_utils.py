import torch
import logging

def get_device():
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device == "cpu":
        logging.warning("Warning: No GPU found.")
    logging.info(f"Device: {_device}")
    return _device