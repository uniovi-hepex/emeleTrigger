"""Hyper-parameter optimisation (HPO) and lightweight NAS helpers."""
from importlib import metadata as _md

__all__ = ["__version__"]
try:
    __version__ = _md.version(__name__)
except _md.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
