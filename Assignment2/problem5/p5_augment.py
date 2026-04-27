"""Compatibility wrapper for importing symbols from 2_augment.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_augment_path = Path(__file__).with_name("2_augment.py")
_spec = spec_from_file_location("p5_augment_impl", _augment_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load augment module from {_augment_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

augment_one = _module.augment_one
generate_augmented = _module.generate_augmented
get_augmented_data = _module.get_augmented_data

__all__ = [
    "augment_one",
    "generate_augmented",
    "get_augmented_data",
]
