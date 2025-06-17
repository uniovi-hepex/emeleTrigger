"""Geometry & housekeeping helpers used by :pymod:`gnn_omtf.data.dataset`.

Everything here is *pure Python* and dependency-light so it can be
unit-tested without ROOT / CUDA.  Numerical heavy-lifting lives elsewhere.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

__all__ = [
    # public constants
    "NUM_PROCESSORS",
    "NUM_PHI_BINS",
    "HW_ETA_TO_ETA_FACTOR",
    "LOGIC_LAYERS_LABEL_MAP",
    # geometry helpers
    "foldPhi",
    "phiZero",
    "stubPhiToGlobalPhi",
    "globalPhiToStubPhi",
    "get_global_phi",
    "get_stub_r",
    # graph-building helpers
    "getEdgesFromLogicLayer",
    "getListOfConnectedLayers",
    "remove_empty_or_nan_graphs",
]

# --------------------------------------------------------------------------- #
# Constants & maps
# --------------------------------------------------------------------------- #

NUM_PROCESSORS: int = 3
NUM_PHI_BINS: int = 5_400
HW_ETA_TO_ETA_FACTOR: float = 0.010875

# Labels for logging / plotting ­– not used in the math itself
LOGIC_LAYERS_LABEL_MAP = {
    0: "MB1",
    2: "MB2",
    4: "MB3",
    6: "ME1/3",
    7: "ME2/2",
    8: "ME3/2",
    9: "ME1/2",
    10: "RB1in",
    11: "RB1out",
    12: "RB2in",
    13: "RB2out",
    14: "RB3",
    15: "RE1/3",
    16: "RE2/3",
    17: "RE3/3",
}

# --------------------------------------------------------------------------- #
# φ helpers
# --------------------------------------------------------------------------- #


def foldPhi(phi: float | int) -> float:
    """Bring φ index into the range [-NUM_PHI_BINS/2, +NUM_PHI_BINS/2)."""
    if phi > NUM_PHI_BINS / 2:
        return phi - NUM_PHI_BINS
    if phi < -NUM_PHI_BINS / 2:
        return phi + NUM_PHI_BINS
    return phi


def phiZero(processor: int) -> float:
    """Local zero for *processor* in hardware bins."""
    return foldPhi(NUM_PHI_BINS / NUM_PROCESSORS * processor + NUM_PHI_BINS / 24)


def stubPhiToGlobalPhi(stubPhi: float, phi_zero: float) -> float:
    """Convert stub φ (hw units) → global φ (rad)."""
    return foldPhi(stubPhi + phi_zero) * (2 * math.pi / NUM_PHI_BINS)


def globalPhiToStubPhi(global_phi: float, phi_zero: float) -> float:
    """Convert global φ (rad) → stub φ (hw units)."""
    return foldPhi(global_phi / (2 * math.pi) * NUM_PHI_BINS - phi_zero)


def get_global_phi(phi, processor: int):
    """Vector-ised convenience wrapper used in the dataset builder."""
    p1phiLSB = 2 * math.pi / NUM_PHI_BINS
    if isinstance(phi, list):
        return [(processor * 192 + p + 600) % NUM_PHI_BINS * p1phiLSB for p in phi]
    return (processor * 192 + phi + 600) % NUM_PHI_BINS * p1phiLSB


# --------------------------------------------------------------------------- #
# r / η helpers
# --------------------------------------------------------------------------- #


def get_stub_r(
    stubTypes, stubEta, stubLayer, stubQuality, *, _eta_factor: float = HW_ETA_TO_ETA_FACTOR
):
    """Return an *array* of radial positions [cm] for each stub in the event.

    Ported 1-to-1 from your original script; cleaned for PEP-8 & numpy.
    """
    rs: list[float] = []
    for stype, eta, layer, qual in zip(
        stubTypes, stubEta, stubLayer, stubQuality, strict=True
    ):
        r: float | None = None

        # ---------- DT stubs ------------------------------------------------
        if stype == 3:
            if layer == 0:
                r = 431.133
            elif layer == 2:
                r = 512.401
            elif layer == 4:
                r = 617.946

            # quality-dependent shift (±23.5/2 cm)
            if qual in {0, 2}:
                r -= 23.5 / 2
            elif qual in {1, 3}:
                r += 23.5 / 2

        # ---------- CSC stubs ----------------------------------------------
        elif stype == 9:
            if layer == 6:
                z = 690
            elif layer == 9:
                z = 700
            elif layer == 7:
                z = 830
            elif layer == 8:
                z = 930
            else:
                raise ValueError(f"Unknown CSC layer {layer}")
            r = z / np.cos(np.tan(2 * np.arctan(np.exp(-eta * _eta_factor))))

        # ---------- RPC stubs ----------------------------------------------
        elif stype == 5:
            if layer == 10:
                r = 413.675
            elif layer == 11:
                r = 448.675
            elif layer == 12:
                r = 494.975
            elif layer == 13:
                r = 529.975
            elif layer == 14:
                r = 602.150
            else:  # end-cap RPC, derive from z
                if layer == 15:
                    z = 720
                elif layer == 16:
                    z = 790
                elif layer == 17:
                    z = 970
                else:
                    raise ValueError(f"Unknown RPC layer {layer}")
                r = z / np.cos(np.tan(2 * np.arctan(np.exp(-eta * _eta_factor))))
        else:
            raise ValueError(f"Unsupported stubType {stype}")

        rs.append(r)

    return np.asarray(rs, dtype=float)


# --------------------------------------------------------------------------- #
# Layer-connection helpers (edge masks)
# --------------------------------------------------------------------------- #


def getEtaKey(eta: float) -> int:
    abs_eta = abs(eta)
    if abs_eta < 0.92:
        return 1
    if abs_eta < 1.1:
        return 2
    if abs_eta < 1.15:
        return 3
    if abs_eta < 1.19:
        return 4
    return 5


def getListOfConnectedLayers(eta: float):
    etaKey = getEtaKey(eta)
    LAYER_ORDER_MAP = {
        1: [10, 0, 11, 12, 2, 13, 14, 4, 6, 15],
        2: [10, 0, 11, 12, 2, 13, 6, 15, 16, 7],
        3: [10, 0, 11, 6, 15, 16, 7, 8, 17],
        4: [10, 0, 11, 16, 7, 8, 17],
        5: [10, 0, 9, 16, 7, 8, 17],
    }
    return LAYER_ORDER_MAP[etaKey]


def getEdgesFromLogicLayer(logicLayer: int, *, withRPC: bool = True) -> List[int]:
    """Return list of layers that *logicLayer* is allowed to connect to."""
    bare = {
        0: [2, 4, 6, 7, 8, 9],
        2: [4, 6, 7],
        4: [6],
        6: [7, 8],
        7: [8, 9],
        8: [9],
        9: [],
    }
    rpc = {
        0: [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        1: [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        2: [4, 6, 7, 10, 11, 12, 13, 14, 15, 16],
        3: [4, 6, 7, 10, 11, 12, 13, 14, 15, 16],
        4: [6, 10, 11, 12, 13, 14, 15],
        5: [6, 10, 11, 12, 13, 14, 15],
        6: [7, 8, 10, 11, 12, 13, 14, 15, 16, 17],
        7: [8, 9, 10, 11, 15, 16, 17],
        8: [9, 10, 11, 15, 16, 17],
        9: [7, 10, 16, 17],
        10: [11, 12, 13, 14, 15, 16, 17],
        11: [12, 13, 14, 15, 16, 17],
        12: [13, 14, 15, 16],
        13: [14, 15, 16],
        14: [15],
        15: [16, 17],
        16: [17],
        17: [],
    }
    if withRPC:
        return rpc[logicLayer]
    if logicLayer >= 10:
        return []
    return bare[logicLayer]


# --------------------------------------------------------------------------- #
# Simple PyG transform helper
# --------------------------------------------------------------------------- #


def remove_empty_or_nan_graphs(data):
    """Return *None* if the PyG `Data` object is empty or contains NaNs.

    Compatible with `pre_transform` hooks in PyG Datasets.
    """
    if data.x.numel() == 0 or data.edge_index.numel() == 0:
        return None
    if (
        torch.isnan(data.x).any()
        or (data.edge_attr is not None and torch.isnan(data.edge_attr).any())
        or torch.isnan(data.y).any()
    ):
        return None
    return data
