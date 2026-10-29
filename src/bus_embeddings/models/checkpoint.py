"""Loading of the checkpoints written by ``bus_embeddings.cli.train_embeddings``."""
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from bus_embeddings.models.embedding_gnn import build_model


def load_checkpoint(path: str | Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Return the model of a checkpoint (in evaluation mode) and the checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["model_type"] == "xgboost":
        model = checkpoint["model"]
    else:
        model = build_model(
            checkpoint["model_type"],
            checkpoint["hyperparams"],
            link_features_in=checkpoint["link_features_in"],
            stop_features_in=checkpoint["stop_features_in"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint
