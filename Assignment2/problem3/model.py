"""Compatibility wrapper for importing symbols from 2_model.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_model_path = Path(__file__).with_name("2_model.py")
_spec = spec_from_file_location("model_impl", _model_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load model module from {_model_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

DigitCNN = _module.DigitCNN
get_model = _module.get_model

__all__ = [
    "DigitCNN",
    "get_model",
]
