"""Compatibility wrapper for importing symbols from 5_utils.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_utils_path = Path(__file__).with_name("5_utils.py")
_spec = spec_from_file_location("utils_impl", _utils_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load utils module from {_utils_path}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

set_seed = _module.set_seed
plot_training_history = _module.plot_training_history
print_summary_table = _module.print_summary_table
save_results_json = _module.save_results_json
compare_variations = _module.compare_variations

__all__ = [
    "set_seed",
    "plot_training_history",
    "print_summary_table",
    "save_results_json",
    "compare_variations",
]
