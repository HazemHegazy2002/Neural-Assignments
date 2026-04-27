"""Compatibility wrapper for importing symbols from 6_experiments.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_experiments_path = Path(__file__).with_name("6_experiments.py")
_spec = spec_from_file_location("p5_experiments_impl", _experiments_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load experiments module from {_experiments_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

run_one = _module.run_one
run_all = _module.run_all
print_table = _module.print_table
load_and_print = _module.load_and_print

__all__ = [
    "run_one",
    "run_all",
    "print_table",
    "load_and_print",
]
