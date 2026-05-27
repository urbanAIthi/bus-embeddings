"""Training and evaluation utilities for GNN embedding models."""
from typing import Any, Dict, List, Tuple
import os
import random

import numpy as np
import torch

from time2model.logging import get_logger
from time2model.training.config import (
    TRAVEL_TIME_RESIDUAL_MODE,
    DWELL_TIME_RESIDUAL_MODE,
    _VALID_TRAVEL_TIME_RESIDUAL_MODES,
    _VALID_DWELL_TIME_RESIDUAL_MODES,
)

logger = get_logger(__name__)


def seed_all(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable fully deterministic CUDA operations (scatter_add, etc.)
    # Required for reproducible PyTorch Geometric message passing
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_geometric_mean(values: List[float]) -> float:
    """Compute geometric mean of non-negative values (for CV fold aggregation)."""
    if not values:
        raise ValueError("Cannot compute geometric mean of empty list")
    if any(v < 0 for v in values):
        raise ValueError(
            f"Geometric mean is undefined for negative values: {values}. "
            f"Use arithmetic mean for metrics that can be negative (e.g., MADGap)."
        )
    
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-15)))))


def get_residual_modes(hyperparams: Dict[str, Any]) -> Tuple[str, str]:
    """Extract and validate travel/dwell time residual modes from hyperparams."""
    tt_mode = hyperparams.get("travel-time-residual-mode", TRAVEL_TIME_RESIDUAL_MODE)
    dt_mode = hyperparams.get("dwell-time-residual-mode", DWELL_TIME_RESIDUAL_MODE)

    if tt_mode not in _VALID_TRAVEL_TIME_RESIDUAL_MODES:
        raise ValueError(
            f"Invalid travel-time-residual-mode='{tt_mode}'. "
            f"Expected one of {sorted(_VALID_TRAVEL_TIME_RESIDUAL_MODES)}"
        )
    if dt_mode not in _VALID_DWELL_TIME_RESIDUAL_MODES:
        raise ValueError(
            f"Invalid dwell-time-residual-mode='{dt_mode}'. "
            f"Expected one of {sorted(_VALID_DWELL_TIME_RESIDUAL_MODES)}"
        )

    return tt_mode, dt_mode

