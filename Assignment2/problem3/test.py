"""Compatibility wrapper for importing symbols from 4_Test.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_test_path = Path(__file__).with_name("4_Test.py")
_spec = spec_from_file_location("test_impl", _test_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load test module from {_test_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

evaluate = _module.evaluate

__all__ = [
    "evaluate",
]
