"""Compatibility wrapper for importing symbols from 1_dataset.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_dataset_path = Path(__file__).with_name("1_dataset.py")
_spec = spec_from_file_location("p5_dataset_impl", _dataset_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load dataset module from {_dataset_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

load_split = _module.load_split
load_test = _module.load_test
prepare_splits = _module.prepare_splits
ReducedMNIST = _module.ReducedMNIST
ROOT_DIR = _module.ROOT_DIR
TRAIN_DIR = _module.TRAIN_DIR
TEST_DIR = _module.TEST_DIR
SPLITS_DIR = _module.SPLITS_DIR
NUM_CLASSES = _module.NUM_CLASSES
SAMPLES_PER_DIGIT_TRAIN = _module.SAMPLES_PER_DIGIT_TRAIN

__all__ = [
    "load_split",
    "load_test",
    "prepare_splits",
    "ReducedMNIST",
    "ROOT_DIR",
    "TRAIN_DIR",
    "TEST_DIR",
    "SPLITS_DIR",
    "NUM_CLASSES",
    "SAMPLES_PER_DIGIT_TRAIN",
]
