"""Training utilities: seeding, fold aggregation and target configuration."""
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from bus_embeddings.training.config import TRAVEL_TIME_RESIDUAL_MODES


def seed_all(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and enable deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuBLAS and scatter operations (PyTorch Geometric message passing)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_geometric_mean(values: List[float]) -> float:
    """Geometric mean of non-negative values (used to aggregate folds)."""
    if not values:
        raise ValueError("Cannot compute geometric mean of empty list")
    if any(v < 0 for v in values):
        raise ValueError(f"Geometric mean is undefined for negative values: {values}")
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-15)))))


def uses_ratio_target(hyperparams: Dict[str, Any]) -> bool:
    """Whether link targets are ratios (True) or differences (False) to the ideal travel time."""
    mode = hyperparams["travel-time-residual-mode"]
    if mode not in TRAVEL_TIME_RESIDUAL_MODES:
        raise ValueError(
            f"Invalid travel-time-residual-mode {mode!r}, "
            f"expected one of {TRAVEL_TIME_RESIDUAL_MODES}"
        )
    return mode == "freeflow_ratio"
