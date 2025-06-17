"""Data-handling sub-package for gnn-omtf.

Exposes the public `OMTFDataset` class:

    from gnn_omtf.data import OMTFDataset
"""
from __future__ import annotations

from .dataset import OMTFDataset  # noqa: F401

__all__ = ["OMTFDataset"]
