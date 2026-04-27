"""Compatibility wrapper for importing symbols from 1_Dataset.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_dataset_path = Path(__file__).with_name("1_Dataset.py")
_spec = spec_from_file_location("dataset_impl", _dataset_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load dataset module from {_dataset_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

TRAIN_DIR = _module.TRAIN_DIR
TEST_DIR = _module.TEST_DIR
SpeechDataset = _module.SpeechDataset
get_datasets = _module.get_datasets

__all__ = [
    "TRAIN_DIR",
    "TEST_DIR",
    "SpeechDataset",
    "get_datasets",
]
