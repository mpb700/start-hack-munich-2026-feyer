"""Differometor — public package entry points.

Heavy, benchmark-irrelevant imports are deliberately deferred:
``visualize_setup`` lives in :mod:`differometor.plot`, which pulls in Plotly.
The benchmark never visualizes anything, so importing Plotly at package import
time would add pure overhead to the measured first-run path.  It is therefore
exposed through a module-level ``__getattr__`` (PEP 562) and only imported on
first use.  ``from differometor import visualize_setup`` keeps working.
"""

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector

__all__ = [
    "Setup",
    "run",
    "run_build_step",
    "run_simulation_step",
    "simulate",
    "power_detector",
    "signal_detector",
    "visualize_setup",
]

_LAZY_ATTRS = {"visualize_setup": "differometor.plot"}


def __getattr__(name: str):
    """Import optional/heavy public attributes only on first access."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
