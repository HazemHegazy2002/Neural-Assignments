"""Compatibility wrapper for importing symbols from 3_Train.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_train_path = Path(__file__).with_name("3_Train.py")
_spec = spec_from_file_location("train_impl", _train_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load train module from {_train_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

train = _module.train

__all__ = [
    "train",
]
