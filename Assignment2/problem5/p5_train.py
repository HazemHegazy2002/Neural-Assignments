"""Compatibility wrapper for importing symbols from 4_train.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_train_path = Path(__file__).with_name("4_train.py")
_spec = spec_from_file_location("p5_train_impl", _train_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load train module from {_train_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

train_model = _module.train_model

__all__ = [
    "train_model",
]
