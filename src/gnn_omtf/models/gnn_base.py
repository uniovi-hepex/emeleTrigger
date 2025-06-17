from __future__ import annotations

"""
Common scaffolding for all GNN architectures.

• BaseGNN – abstract super-class adding logging helpers,
  parameter counting and freeze/unfreeze utilities.

• MODEL_REGISTRY – global mapping from lowercase model
  names to the corresponding subclass.

• register_model(name) – decorator that registers in one line:
      @register_model("gat")
      class GATRegressor(BaseGNN): ...
"""


import abc
import logging
from typing import Dict, Type

import torch
from torch import nn

__all__ = ["BaseGNN", "register_model"]

# --------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------- #

MODEL_REGISTRY: Dict[str, Type["BaseGNN"]] = {}


def register_model(name: str):  # noqa: D401 – “Register *name*”
    """Class-decorator that adds *name* → *class* to :data:`MODEL_REGISTRY`."""

    def decorator(cls: Type["BaseGNN"]):
        if name.lower() in MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' already registered.")
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class BaseGNN(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base-class – *all* architectures inherit from this."""

    #: Default dropout to apply between layers (sub-classes may override)
    dropout_p: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ API #

    @abc.abstractmethod
    def forward(self, data: "torch_geometric.data.Data") -> torch.Tensor:  # noqa: D401,E501
        """Compute model output for a single batch passed as a PyG `Data`."""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ convenience tools #

    @staticmethod
    def get(name: str, **kwargs) -> "BaseGNN":
        """Factory method: `BaseGNN.get("gat", **cfg)`."""
        key = name.lower()
        try:
            cls = MODEL_REGISTRY[key]
        except KeyError as err:  # pragma: no cover – tested indirectly
            raise KeyError(
                f"Unknown model '{name}'. "
                f"Available: {', '.join(MODEL_REGISTRY)}"
            ) from err
        return cls(**kwargs)  # type: ignore[arg-type]

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return number of (trainable) parameters."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ freeze helpers #

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
