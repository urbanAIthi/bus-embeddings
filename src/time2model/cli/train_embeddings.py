"""Train GNN embeddings (GraphSAGE / GAT) via the proxy task: predict raw
travel/dwell times from static graph features and a time context vector.

Municipalities are split train/val/test by category (large city, small city,
town, rural). Each fold uses an expanding train window and fixed-size val/test
windows.
"""
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import wandb
from torch_geometric.data import HeteroData

from time2model.utils.splits import generate_split_boundaries

from time2model.logging import get_logger, setup_logging
from time2model.models.embedding_gnn import GraphSAGEEmbedding, GATEmbedding
from time2model.models.xgboost_baseline import XGBoostBaseline
from time2model.data.loaders import get_time_range
from time2model.data.proxy_task_dataset import (
    GlobalScaler,
    ProxyTaskBatchIterator,
    RawObservationProxyDataset,
)
from time2model.utils.hetero_graph_utils import build_hetero_graph
from time2model.training.train_utils import seed_all, compute_geometric_mean, get_residual_modes
from time2model.training.metrics import compute_oversmoothing_metrics, compute_seasonal_naive_metrics, compute_smase_by_region, compute_smape, compute_node_level_metrics
from time2model.training.config import (
    SEED,
    LOG_INTERVAL,
    N_SPLITS,
    REGION_SPLIT,
    STOP_CLASSIFIER_THRESHOLDS,
    EMBEDDING_GRAPHSAGE_DEFAULTS,
    EMBEDDING_GAT_DEFAULTS,
    EMBEDDING_XGBOOST_DEFAULTS,
    EMBEDDING_MODEL_DISPLAY_NAMES,
)

logger = get_logger(__name__)


# Municipality categories for stratified region splitting.
MUNICIPALITY_CATEGORIES: Dict[str, List[str]] = {
    "large_city": ["GM0599", "GM0518", "GM0344"],
    "small_city": ["GM0546", "GM0503", "GM1930", "GM0590", "GM1842"],
    "town": ["GM0281", "GM0059", "GM0047", "GM0629", "GM0312"],
    "rural": ["GM1969", "GM1731", "GM1950", "GM1681", "GM1690"],
}

MUNICIPALITY_TO_CATEGORY: Dict[str, str] = {}
for _cat, _ids in MUNICIPALITY_CATEGORIES.items():
    for _mid in _ids:
        MUNICIPALITY_TO_CATEGORY[_mid] = _cat

ALL_MUNICIPALITIES: List[str] = sorted(MUNICIPALITY_TO_CATEGORY.keys())

# Target split sizes per category — informational. Actual split lives in
# config.REGION_SPLIT. Large=1/1/1; non-large=3/1/1, 2/2/1, 2/1/2. Total 8/5/5.
CATEGORY_SPLIT_SIZES: Dict[str, Tuple[int, int, int]] = {
    "large_city": (1, 1, 1),
    "small_city": (3, 1, 1),
    "town": (2, 2, 1),
    "rural": (2, 1, 2),
}


def assign_regions_to_splits(
    available_regions: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Return the predefined train/val/test region split from config.

    Raises ValueError if any predefined region is not in available_regions.
    """
    available_set = set(available_regions)
    for split_name, regions in REGION_SPLIT.items():
        missing = [r for r in regions if r not in available_set]
        if missing:
            raise ValueError(
                f"Predefined {split_name} regions {missing} are not available. "
                f"Available regions: {sorted(available_set)}"
            )
    return (
        sorted(REGION_SPLIT["train"]),
        sorted(REGION_SPLIT["val"]),
        sorted(REGION_SPLIT["test"]),
    )


def discover_regions(
    artifacts_dir: str,
    parquet_municipalities: set[str],
) -> List[str]:
    """Auto-discover municipalities that have both artifacts and parquet data."""
    # Find municipalities with artifacts
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.is_dir():
        raise ValueError(f"Artifacts directory does not exist: {artifacts_dir}")

    artifact_regions = set()
    for child in artifacts_path.iterdir():
        if child.is_dir():
            # Check for required graph files
            has_link = (child / "nodes_link.csv").exists()
            has_stop = (child / "nodes_stop.csv").exists()
            has_edges = (child / "edges.csv").exists()
            if has_link and has_stop and has_edges:
                artifact_regions.add(child.name)

    # Intersection: must have artifacts and be in parquet data
    available = sorted(artifact_regions & parquet_municipalities)

    if not available:
        raise ValueError(
            f"No municipalities found with all required data. "
            f"Artifacts: {sorted(artifact_regions)}, "
            f"Parquet municipalities: {sorted(parquet_municipalities)}"
        )

    logger.info(
        "Discovered %d municipalities with complete data: %s",
        len(available),
        available,
    )
    return available


def get_region_artifact_paths(
    regions: List[str],
    artifacts_dir: str,
) -> Tuple[List[str | Path], List[str | Path], List[str | Path]]:
    """Build artifact file paths (link/stop/edge CSVs) for a list of municipalities."""
    link_csvs: List[str | Path] = [os.path.join(artifacts_dir, r, "nodes_link.csv") for r in regions]
    stop_csvs: List[str | Path] = [os.path.join(artifacts_dir, r, "nodes_stop.csv") for r in regions]
    edge_csvs: List[str | Path] = [os.path.join(artifacts_dir, r, "edges.csv") for r in regions]
    return link_csvs, stop_csvs, edge_csvs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN embeddings with proxy task"
    )

    # Directory-based inputs (auto-discover regions)
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Directory containing per-municipality artifact subdirectories "
             "(each with nodes_link.csv, nodes_stop.csv, edges.csv).",
    )
    parser.add_argument(
        "--travel-times-parquet",
        type=str,
        required=True,
        help="Path to preprocessed travel_times.parquet (from preprocess_data).",
    )
    parser.add_argument(
        "--dwell-times-parquet",
        type=str,
        required=True,
        help="Path to preprocessed dwell_times.parquet (from preprocess_data).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="embeddings_out",
        help="Output directory for model weights (default: embeddings_out)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["graphsage", "gat", "xgboost"],
        default="graphsage",
        help="Model to train (default: graphsage). 'xgboost' runs a flat baseline "
             "using the same dataset/feature pipeline but no graph structure.",
    )

    # Training parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )

    # Logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="time2model-embeddings",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level (default: INFO)",
    )
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        help="Filter to a single route by municipality:route (e.g. GM0344:118101). "
             "Only data from that municipality+route combination will be used.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable sweep mode (trains on last split only)",
    )

    args = parser.parse_args()

    # Validate route format if specified
    if args.route is not None:
        if ":" not in args.route:
            raise ValueError(
                f"--route must be in format 'municipality:route' (e.g. GM0344:118101), "
                f"got '{args.route}'"
            )
        route_municipality, route_id = args.route.split(":", 1)
        args.route_municipality = route_municipality
        args.route_id = route_id
    else:
        args.route_municipality = None
        args.route_id = None

    return args


def train_epoch(
    model: GraphSAGEEmbedding | GATEmbedding,
    graph_data: HeteroData,
    train_loader: ProxyTaskBatchIterator,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    link_reg_weight: float = 1.0,
    stop_reg_weight: float = 1.0,
    stop_cls_weight: float = 1.0,
    grad_clip: float = 1.0,
    scaler: GradScaler | None = None,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    focal_gamma: float = 0.0,
    scheduler: CosineAnnealingWarmRestarts | OneCycleLR | None = None,
) -> tuple[float, float, float, float, float]:
    """Train one epoch. Returns (combined, link, stop_reg, stop_cls, stop_combined) losses."""
    model.train()
    # Accumulate losses on GPU to avoid sync overhead.
    total_link_loss = torch.tensor(0.0, device=device)
    total_stop_reg_loss = torch.tensor(0.0, device=device)
    total_stop_cls_loss = torch.tensor(0.0, device=device)
    total_link_samples = 0
    total_stop_samples = 0
    total_stop_positive_samples = 0

    use_amp = scaler is not None and device.type == "cuda"
    graph_data = graph_data.to(device)

    if loss_type == "huber":
        reg_loss_fn = lambda pred, target: F.huber_loss(pred, target, reduction='mean', delta=huber_delta)
    elif loss_type == "l1":
        reg_loss_fn = lambda pred, target: F.l1_loss(pred, target, reduction='mean')
    elif loss_type == "logcosh":
        def _logcosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            diff = pred - target
            return torch.mean(diff + F.softplus(-2.0 * diff, beta=1) - 0.6931471805599453)
        reg_loss_fn = _logcosh_loss
    else:
        reg_loss_fn = lambda pred, target: F.mse_loss(pred, target, reduction='mean')

    n_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        link_indices, link_ctx, link_targets, stop_indices, stop_ctx, stop_targets, stop_rate_targets = batch

        link_indices = link_indices.to(device)
        link_ctx = link_ctx.to(device)
        link_targets = link_targets.to(device)
        stop_indices = stop_indices.to(device)
        stop_ctx = stop_ctx.to(device)
        stop_targets = stop_targets.to(device)
        if stop_rate_targets is not None:
            stop_rate_targets = stop_rate_targets.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=use_amp):
            link_preds, stop_preds, stop_logits, embeddings = model(
                graph_data,
                link_indices, link_ctx,
                stop_indices, stop_ctx,
            )

            link_loss = torch.tensor(0.0, device=device)
            if link_preds.numel() > 0:
                link_loss = reg_loss_fn(link_preds, link_targets)

            stop_reg_loss = torch.tensor(0.0, device=device)
            stop_cls_loss = torch.tensor(0.0, device=device)

            if stop_preds.numel() > 0 and stop_rate_targets is not None:
                # Classification: focal BCE for P(will stop).
                bce = F.binary_cross_entropy_with_logits(
                    stop_logits, stop_rate_targets, reduction='none'
                )
                if focal_gamma > 0:
                    p_t = torch.exp(-bce)
                    stop_cls_loss = ((1 - p_t) ** focal_gamma * bce).mean()
                else:
                    stop_cls_loss = bce.mean()

                # Dwell regression only over stopped samples.
                stopped_mask = stop_rate_targets > 0.5
                if stopped_mask.any():
                    stop_reg_loss = reg_loss_fn(
                        stop_preds[stopped_mask],
                        stop_targets[stopped_mask],
                    )

            _stop_combined_loss = stop_cls_weight * stop_cls_loss + stop_reg_weight * stop_reg_loss

            combined_loss = (
                link_reg_weight * link_loss + stop_cls_weight * stop_cls_loss + stop_reg_weight * stop_reg_loss
            ) / (link_reg_weight + stop_cls_weight + stop_reg_weight)

        if use_amp and scaler is not None:
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(scheduler.last_epoch + (batch_idx + 1) / n_batches)
            elif isinstance(scheduler, OneCycleLR):
                scheduler.step()

        total_link_loss += link_loss.detach() * link_preds.numel()
        total_link_samples += link_preds.numel()

        if stop_preds.numel() > 0 and stop_rate_targets is not None:
            total_stop_cls_loss += stop_cls_loss.detach() * stop_preds.numel()
            total_stop_samples += stop_preds.numel()
            stopped_mask = stop_rate_targets > 0.5
            if stopped_mask.any():
                total_stop_reg_loss += stop_reg_loss.detach() * stopped_mask.sum()
                total_stop_positive_samples += stopped_mask.sum().item()

    avg_link_loss = (total_link_loss / max(total_link_samples, 1)).item()
    avg_stop_reg_loss = (total_stop_reg_loss / max(total_stop_positive_samples, 1)).item()
    avg_stop_cls_loss = (total_stop_cls_loss / max(total_stop_samples, 1)).item()
    avg_stop_combined = stop_cls_weight * avg_stop_cls_loss + stop_reg_weight * avg_stop_reg_loss
    avg_combined = (
        link_reg_weight * avg_link_loss + stop_cls_weight * avg_stop_cls_loss + stop_reg_weight * avg_stop_reg_loss
    ) / (link_reg_weight + stop_cls_weight + stop_reg_weight)

    return avg_combined, avg_link_loss, avg_stop_reg_loss, avg_stop_cls_loss, avg_stop_combined


def evaluate(
    model: GraphSAGEEmbedding | GATEmbedding,
    graph_data: HeteroData,
    val_loader: ProxyTaskBatchIterator,
    device: torch.device,
    link_reg_weight: float = 1.0,
    stop_reg_weight: float = 1.0,
    stop_cls_weight: float = 1.0,
    use_amp: bool = False,
    link_scaler: "GlobalScaler | None" = None,
    stop_scaler: "GlobalScaler | None" = None,
    prefix: str = "val",
    log_transform: bool = False,
    link_freeflow_refs: "torch.Tensor | None" = None,
    freeflow_ratio: bool = False,
    link_municipalities: "np.ndarray | None" = None,
    stop_municipalities: "np.ndarray | None" = None,
) -> tuple[float, float, float, float, float, Dict[str, Any]]:
    """Evaluate on val/test. Returns losses and a metrics dict.

    When ``link_freeflow_refs`` is provided, ``val_loader`` MUST iterate in
    sequential order (``shuffle=False``) so concatenated batch predictions
    align positionally with the per-sample freeflow reference tensor.
    """
    model.eval()
    total_link_loss = 0.0
    total_stop_reg_loss = 0.0
    total_stop_cls_loss = 0.0
    total_link_samples = 0
    total_stop_samples = 0
    total_stop_positive_samples = 0

    link_preds_all = []
    link_targets_all = []
    link_indices_all = []
    link_ctx_all = []
    stop_preds_all = []
    stop_targets_all = []
    stop_indices_all = []
    stop_ctx_all = []
    stop_logits_all = []
    stop_rate_all = []

    graph_data = graph_data.to(device)
    amp_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        for batch in val_loader:
            link_indices, link_ctx, link_targets, stop_indices, stop_ctx, stop_targets, stop_rate_targets = batch

            link_indices = link_indices.to(device)
            link_ctx = link_ctx.to(device)
            link_targets = link_targets.to(device)
            stop_indices = stop_indices.to(device)
            stop_ctx = stop_ctx.to(device)
            stop_targets = stop_targets.to(device)
            if stop_rate_targets is not None:
                stop_rate_targets = stop_rate_targets.to(device)

            with autocast(device_type="cuda", enabled=amp_enabled):
                link_preds, stop_preds, stop_logits, _ = model(
                    graph_data,
                    link_indices, link_ctx,
                    stop_indices, stop_ctx,
                )

            if amp_enabled:
                link_preds = link_preds.float()
                stop_preds = stop_preds.float()
                stop_logits = stop_logits.float()

            if link_preds.numel() > 0:
                link_loss = F.mse_loss(link_preds, link_targets, reduction='sum')
                total_link_loss += link_loss.item()
                total_link_samples += link_preds.numel()
                link_preds_all.append(link_preds.cpu())
                link_targets_all.append(link_targets.cpu())
                link_indices_all.append(link_indices.cpu())
                link_ctx_all.append(link_ctx.cpu())

            if stop_preds.numel() > 0 and stop_rate_targets is not None:
                stop_cls_loss = F.binary_cross_entropy_with_logits(
                    stop_logits, stop_rate_targets, reduction='sum'
                )
                total_stop_cls_loss += stop_cls_loss.item()
                total_stop_samples += stop_preds.numel()

                stopped_mask = stop_rate_targets > 0.5
                if stopped_mask.any():
                    stop_reg_loss = F.mse_loss(
                        stop_preds[stopped_mask],
                        stop_targets[stopped_mask],
                        reduction='sum',
                    )
                    total_stop_reg_loss += stop_reg_loss.item()
                    total_stop_positive_samples += stopped_mask.sum().item()

                # Regression metrics use only stopped samples.
                stop_preds_all.append(stop_preds[stopped_mask].cpu())
                stop_targets_all.append(stop_targets[stopped_mask].cpu())
                stop_indices_all.append(stop_indices[stopped_mask].cpu())
                stop_ctx_all.append(stop_ctx[stopped_mask].cpu())
                stop_logits_all.append(stop_logits.cpu())
                stop_rate_all.append(stop_rate_targets.cpu())

    avg_link_loss = total_link_loss / max(total_link_samples, 1)
    avg_stop_reg_loss = total_stop_reg_loss / max(total_stop_positive_samples, 1)
    avg_stop_cls_loss = total_stop_cls_loss / max(total_stop_samples, 1)
    avg_stop_combined = stop_cls_weight * avg_stop_cls_loss + stop_reg_weight * avg_stop_reg_loss
    avg_combined = (
        link_reg_weight * avg_link_loss + stop_cls_weight * avg_stop_cls_loss + stop_reg_weight * avg_stop_reg_loss
    ) / (link_reg_weight + stop_cls_weight + stop_reg_weight)

    # Compute regression metrics (MAE, RMSE in original scale)
    metrics_dict: Dict[str, Any] = {}

    if link_preds_all and link_scaler is not None:
        all_link_preds = torch.cat(link_preds_all, dim=0)
        all_link_targets = torch.cat(link_targets_all, dim=0)

        link_preds_orig = link_scaler.inverse_transform_tensor(all_link_preds)
        link_targets_orig = link_scaler.inverse_transform_tensor(all_link_targets)

        # Undo residual/ratio + log transforms to recover absolute seconds.
        if link_freeflow_refs is not None:
            if len(link_preds_orig) != len(link_freeflow_refs):
                raise RuntimeError(
                    f"Freeflow refs length mismatch: {len(link_preds_orig)} predictions "
                    f"vs {len(link_freeflow_refs)} refs. "
                    f"Ensure the evaluation DataLoader uses shuffle=False."
                )
            if freeflow_ratio:
                link_preds_orig = link_preds_orig * link_freeflow_refs
                link_targets_orig = link_targets_orig * link_freeflow_refs
            else:
                link_preds_orig = link_preds_orig + link_freeflow_refs
                link_targets_orig = link_targets_orig + link_freeflow_refs

        if log_transform:
            link_preds_orig = torch.expm1(link_preds_orig)
            link_targets_orig = torch.expm1(link_targets_orig)

        link_preds_orig = link_preds_orig.clamp(min=0.0)

        link_mae = (link_preds_orig - link_targets_orig).abs().mean().item()
        link_rmse = ((link_preds_orig - link_targets_orig) ** 2).mean().sqrt().item()
        link_bias = (link_preds_orig - link_targets_orig).mean().item()
        link_smape = compute_smape(link_preds_orig, link_targets_orig)

        metrics_dict[f"{prefix}/link/mae"] = link_mae
        metrics_dict[f"{prefix}/link/rmse"] = link_rmse
        metrics_dict[f"{prefix}/link/bias"] = link_bias
        metrics_dict[f"{prefix}/link/smape"] = link_smape

        # Stash raw predictions for per-region sMASE/sRMSSE in _compute_smase_metrics.
        metrics_dict[f"{prefix}/_link_preds_orig"] = link_preds_orig.numpy()
        metrics_dict[f"{prefix}/_link_targets_orig"] = link_targets_orig.numpy()
        if link_municipalities is not None:
            metrics_dict[f"{prefix}/_link_municipalities"] = link_municipalities

        if link_indices_all:
            all_link_indices = torch.cat(link_indices_all, dim=0)
            all_link_ctx = torch.cat(link_ctx_all, dim=0) if link_ctx_all else None
            link_node_metrics = compute_node_level_metrics(
                link_preds_orig, link_targets_orig, all_link_indices, all_link_ctx,
            )
            metrics_dict[f"{prefix}/link/node_r2"] = link_node_metrics["node_r2"]
            metrics_dict[f"{prefix}/link/node_mape"] = link_node_metrics["node_mape"]
            metrics_dict[f"{prefix}/link/node_mae"] = link_node_metrics["node_mae"]
            metrics_dict[f"{prefix}/link/node_smape"] = link_node_metrics["node_smape"]
            metrics_dict[f"{prefix}/link/node_count"] = link_node_metrics["node_count"]

    if stop_preds_all and stop_scaler is not None:
        all_stop_preds = torch.cat(stop_preds_all, dim=0)
        all_stop_targets = torch.cat(stop_targets_all, dim=0)

        stop_preds_orig = stop_scaler.inverse_transform_tensor(all_stop_preds)
        stop_targets_orig = stop_scaler.inverse_transform_tensor(all_stop_targets)

        if log_transform:
            stop_preds_orig = torch.expm1(stop_preds_orig)
            stop_targets_orig = torch.expm1(stop_targets_orig)

        stop_mae = (stop_preds_orig - stop_targets_orig).abs().mean().item()
        stop_rmse = ((stop_preds_orig - stop_targets_orig) ** 2).mean().sqrt().item()
        stop_smape = compute_smape(stop_preds_orig, stop_targets_orig)

        metrics_dict[f"{prefix}/stop/mae"] = stop_mae
        metrics_dict[f"{prefix}/stop/rmse"] = stop_rmse
        metrics_dict[f"{prefix}/stop/smape"] = stop_smape

        metrics_dict[f"{prefix}/_stop_preds_orig"] = stop_preds_orig.numpy()
        metrics_dict[f"{prefix}/_stop_targets_orig"] = stop_targets_orig.numpy()
        if stop_municipalities is not None:
            metrics_dict[f"{prefix}/_stop_municipalities"] = stop_municipalities

        if stop_indices_all:
            all_stop_indices = torch.cat(stop_indices_all, dim=0)
            all_stop_ctx = torch.cat(stop_ctx_all, dim=0) if stop_ctx_all else None
            stop_node_metrics = compute_node_level_metrics(
                stop_preds_orig, stop_targets_orig, all_stop_indices, all_stop_ctx,
            )
            metrics_dict[f"{prefix}/stop/node_r2"] = stop_node_metrics["node_r2"]
            metrics_dict[f"{prefix}/stop/node_mape"] = stop_node_metrics["node_mape"]
            metrics_dict[f"{prefix}/stop/node_mae"] = stop_node_metrics["node_mae"]
            metrics_dict[f"{prefix}/stop/node_smape"] = stop_node_metrics["node_smape"]
            metrics_dict[f"{prefix}/stop/node_count"] = stop_node_metrics["node_count"]

    # Classification metrics for stops (binary stopped vs pass-through).
    if stop_logits_all and stop_rate_all:
        all_stop_logits = torch.cat(stop_logits_all, dim=0)
        all_stop_labels = torch.cat(stop_rate_all, dim=0)
        stop_probs = torch.sigmoid(all_stop_logits)

        pct_stopped = all_stop_labels.mean().item()
        metrics_dict[f"{prefix}/stop_pct_stopped"] = pct_stopped

        # Per-model threshold from STOP_CLASSIFIER_THRESHOLDS keeps F1/accuracy
        # comparable across architectures despite their different calibrations.
        _model_type = getattr(model, "MODEL_TYPE", None)
        _threshold = STOP_CLASSIFIER_THRESHOLDS.get(_model_type, 0.5)
        metrics_dict[f"{prefix}/stop_threshold"] = _threshold
        stop_preds_binary = (stop_probs > _threshold).float()
        accuracy = (stop_preds_binary == all_stop_labels).float().mean().item()
        metrics_dict[f"{prefix}/stop_accuracy"] = accuracy

        true_positives = ((stop_preds_binary == 1) & (all_stop_labels == 1)).sum().float()
        predicted_positives = (stop_preds_binary == 1).sum().float()
        actual_positives = (all_stop_labels == 1).sum().float()

        precision = (true_positives / predicted_positives).item() if predicted_positives > 0 else 0.0
        recall = (true_positives / actual_positives).item() if actual_positives > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_dict[f"{prefix}/stop_precision"] = precision
        metrics_dict[f"{prefix}/stop_recall"] = recall
        metrics_dict[f"{prefix}/stop_f1"] = f1

        brier_score = ((stop_probs - all_stop_labels) ** 2).mean().item()
        metrics_dict[f"{prefix}/stop_brier"] = brier_score

        try:
            from sklearn.metrics import roc_auc_score
            labels_np = all_stop_labels.numpy()
            probs_np = stop_probs.numpy()
            if len(np.unique(labels_np)) > 1:
                auc_roc = roc_auc_score(labels_np, probs_np)
                metrics_dict[f"{prefix}/stop_auc"] = auc_roc
        except ImportError:
            pass

    return avg_combined, avg_link_loss, avg_stop_reg_loss, avg_stop_cls_loss, avg_stop_combined, metrics_dict


def split_metrics_by_node_type(metrics: Dict[str, Any], target_prefix: str, best_prefix: bool = False) -> Dict[str, Any]:
    """Reorganize metrics dict by node type for W&B logging."""
    result = {}
    prefix = "best_" if best_prefix else ""
    for k, v in metrics.items():
        # Extract metric name after node type
        if "/link/" in k:
            metric_name = k.split("/link/")[-1]
            result[f"{prefix}{target_prefix}_link/{metric_name}"] = v
        elif "/stop" in k:
            # Handles both "/stop/" and "/stop_" patterns
            metric_name = k.split("/stop")[-1].lstrip("_/")
            result[f"{prefix}{target_prefix}_stop/{metric_name}"] = v
    return result


def split_oversmoothing_metrics(metrics: Dict[str, Any], best_prefix: bool = False) -> Dict[str, Any]:
    """Reorganize oversmoothing metrics by node type for W&B logging."""
    result = {}
    prefix = "best_" if best_prefix else ""
    for k, v in metrics.items():
        if "link/" in k:
            metric_name = k.split("link/")[-1]
            result[f"{prefix}oversmoothing_link/{metric_name}"] = v
        elif "stop/" in k:
            metric_name = k.split("stop/")[-1]
            result[f"{prefix}oversmoothing_stop/{metric_name}"] = v
    return result


def _make_loader(
    dataset: "RawObservationProxyDataset",
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    subsample: float = 1.0,
) -> ProxyTaskBatchIterator:
    """Create a batch iterator with standard settings."""
    return ProxyTaskBatchIterator(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
        subsample=subsample,
    )


def _compute_smase_metrics(
    df_eval: pd.DataFrame,
    df_history: pd.DataFrame,
    id_col: str,
    model_mae: float,
    model_rmse: float,
    metrics_dict: Dict[str, Any],
    prefix: str,
) -> Tuple[str, str]:
    """Compute sMASE/sRMSSE, merge into metrics_dict, return formatted strings.

    Prefers the per-region geometric mean when raw predictions and municipality
    labels are present (stored by ``evaluate``); falls back to a pooled
    computation otherwise.
    """
    eval_prefix = prefix.rsplit("/", 1)[0]
    preds_key = f"{eval_prefix}/_{id_col}_preds_orig"
    targets_key = f"{eval_prefix}/_{id_col}_targets_orig"
    muni_key = f"{eval_prefix}/_{id_col}_municipalities"

    use_per_region = (
        "municipality" in df_eval.columns
        and preds_key in metrics_dict
        and targets_key in metrics_dict
        and muni_key in metrics_dict
    )

    if use_per_region:
        sn = compute_smase_by_region(
            df_eval=df_eval,
            df_history=df_history,
            id_col=id_col,
            time_col="from_time",
            value_col="travel_time",
            region_col="municipality",
            model_preds_orig=metrics_dict[preds_key],
            model_targets_orig=metrics_dict[targets_key],
            model_municipalities=metrics_dict[muni_key],
        )
    else:
        sn = compute_seasonal_naive_metrics(
            df_eval=df_eval,
            df_history=df_history,
            id_col=id_col,
            time_col="from_time",
            value_col="travel_time",
            model_mae=model_mae,
            model_rmse=model_rmse,
        )

    for k, v in sn.items():
        if not isinstance(v, dict):  # skip per_region diagnostic dict
            metrics_dict[f"{prefix}/{k}"] = v
    smase_str = f"{sn['smase']:6.3f}" if "smase" in sn else "   N/A"
    srmsse_str = f"{sn['srmsse']:6.3f}" if "srmsse" in sn else "   N/A"
    return smase_str, srmsse_str



def _train_xgboost_split(
    args: argparse.Namespace,
    hyperparams: Dict[str, Any],
    device: torch.device,
    train_graph: HeteroData,
    val_graph: Optional[HeteroData],
    train_dataset: "RawObservationProxyDataset",
    val_dataset: Optional["RawObservationProxyDataset"],
    val_loader: Optional[ProxyTaskBatchIterator],
    df_travel_val: pd.DataFrame,
    df_travel_val_full: pd.DataFrame,
    df_dwell_val: pd.DataFrame,
    df_dwell_val_full: pd.DataFrame,
    log_transform: bool,
    freeflow_ratio: bool,
    run: Optional[Any] = None,
) -> Tuple[XGBoostBaseline, float, Optional[Dict[str, Any]], Optional[Dict[str, float]]]:
    """Fit the XGBoost baseline and run a single val-transfer evaluation.

    XGBoost handles its own early stopping via the val set, so the outer epoch
    loop and oversmoothing tracking used by the GNN path are skipped.
    """
    model = XGBoostBaseline(
        n_estimators=hyperparams["xgb-n-estimators"],
        max_depth=hyperparams["xgb-max-depth"],
        learning_rate=hyperparams["xgb-learning-rate"],
        subsample=hyperparams["xgb-subsample"],
        colsample_bytree=hyperparams["xgb-colsample-bytree"],
        min_child_weight=hyperparams["xgb-min-child-weight"],
        early_stopping_rounds=hyperparams["xgb-early-stopping-rounds"],
        random_state=args.seed,
    ).to(device)
    model.fit(train_dataset, train_graph, val_dataset, val_graph)

    best_val_metrics: Optional[Dict[str, Any]] = None
    best_val_loss: float = 0.0

    if val_loader is None or val_graph is None or val_dataset is None:
        return model, best_val_loss, best_val_metrics, None

    _, _, _, _, _, val_metrics = evaluate(
        model, val_graph, val_loader, device,
        link_reg_weight=hyperparams.get("link-reg-weight", 1.0),
        stop_reg_weight=hyperparams.get("stop-reg-weight", 1.0),
        stop_cls_weight=hyperparams.get("stop-cls-weight", 1.0),
        use_amp=False,
        link_scaler=train_dataset.link_scaler,
        stop_scaler=train_dataset.stop_scaler,
        log_transform=log_transform,
        link_freeflow_refs=val_dataset.link_freeflow_refs,
        freeflow_ratio=freeflow_ratio,
        link_municipalities=val_dataset.link_municipalities,
        stop_municipalities=val_dataset.stop_municipalities,
    )

    link_mae = val_metrics.get("val/link/mae", 0)
    link_rmse = val_metrics.get("val/link/rmse", 0)
    stop_mae = val_metrics.get("val/stop/mae", 0)
    stop_rmse = val_metrics.get("val/stop/rmse", 0)

    link_smase_str, link_srmsse_str = "   N/A", "   N/A"
    if link_mae > 0:
        link_smase_str, link_srmsse_str = _compute_smase_metrics(
            df_travel_val, df_travel_val_full, "link",
            link_mae, link_rmse, val_metrics, "val/link",
        )
    stop_smase_str, stop_srmsse_str = "   N/A", "   N/A"
    if stop_mae > 0:
        stop_smase_str, stop_srmsse_str = _compute_smase_metrics(
            df_dwell_val[df_dwell_val["travel_time"] > 0],
            df_dwell_val_full[df_dwell_val_full["travel_time"] > 0],
            "stop", stop_mae, stop_rmse, val_metrics, "val/stop",
        )

    link_smape = val_metrics.get("val/link/smape", 0)
    stop_smape = val_metrics.get("val/stop/smape", 0)
    stop_f1 = val_metrics.get("val/stop_f1", 0)
    stop_auc = val_metrics.get("val/stop_auc", 0)
    stop_accuracy = val_metrics.get("val/stop_accuracy", 0)

    link_srmsse = val_metrics.get("val/link/srmsse")
    stop_srmsse = val_metrics.get("val/stop/srmsse")
    if link_srmsse is not None and stop_srmsse is not None:
        best_val_loss = (link_srmsse * stop_srmsse) ** 0.5

    best_val_metrics = dict(val_metrics)
    logger.info("XGBoost fit complete | early_stop_metric=%.4f", best_val_loss)
    logger.info(
        "  Link  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s",
        link_mae, link_rmse, link_smape, link_smase_str, link_srmsse_str,
    )
    logger.info(
        "  Stop  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s | F1: %6.3f | AUC: %6.3f | Acc: %6.3f",
        stop_mae, stop_rmse, stop_smape, stop_smase_str, stop_srmsse_str,
        stop_f1, stop_auc, stop_accuracy,
    )

    if run is not None:
        log_dict = split_metrics_by_node_type(val_metrics, "val_transfer")
        link_smase_wb = log_dict.get("val_transfer_link/smase")
        stop_smase_wb = log_dict.get("val_transfer_stop/smase")
        if link_smase_wb is not None and stop_smase_wb is not None:
            log_dict["val_transfer_combined/smase"] = (link_smase_wb * stop_smase_wb) ** 0.5
        link_srmsse_wb = log_dict.get("val_transfer_link/srmsse")
        stop_srmsse_wb = log_dict.get("val_transfer_stop/srmsse")
        if link_srmsse_wb is not None and stop_srmsse_wb is not None:
            log_dict["val_transfer_combined/srmsse"] = (link_srmsse_wb * stop_srmsse_wb) ** 0.5
        run.log(log_dict)

    return model, best_val_loss, best_val_metrics, None


def _train_gnn_split(
    args: argparse.Namespace,
    hyperparams: Dict[str, Any],
    device: torch.device,
    train_graph: HeteroData,
    val_graph: Optional[HeteroData],
    train_regions: List[str],
    train_dataset: "RawObservationProxyDataset",
    val_dataset: Optional["RawObservationProxyDataset"],
    train_valtime_dataset: Optional["RawObservationProxyDataset"],
    train_loader: ProxyTaskBatchIterator,
    val_loader: Optional[ProxyTaskBatchIterator],
    train_valtime_loader: Optional[ProxyTaskBatchIterator],
    df_travel_val: pd.DataFrame,
    df_travel_val_full: pd.DataFrame,
    df_dwell_val: pd.DataFrame,
    df_dwell_val_full: pd.DataFrame,
    log_transform: bool,
    freeflow_ratio: bool,
    run: Optional[Any],
) -> Tuple[Any, float, Optional[Dict[str, Any]], Optional[Dict[str, float]]]:
    """Train one GraphSAGE / GAT fold and return the best model + val metrics."""
    # Initialize model (dimensions from train graph)
    link_features = train_graph["link"].x.size(-1)
    stop_features = train_graph["stop"].x.size(-1)

    if args.model == "graphsage":
        model = GraphSAGEEmbedding(
            link_features_in=link_features,
            stop_features_in=stop_features,
            gnn_hidden_dim=hyperparams["gnn-hidden-dim"],
            gnn_output_dim=hyperparams["gnn-output-dim"],
            gnn_layers=hyperparams["gnn-layers"],
            skip_connections=hyperparams.get("skip-connections", True),
            global_skip_connection=hyperparams.get("global-skip-connection", False),
            dropout=hyperparams["dropout"],
            head_hidden_dim=hyperparams["head-hidden-dim"],
            head_type=hyperparams.get("head-type", "film"),
            head_time_concat=hyperparams.get("head-time-concat", True),
            n_head_layers=hyperparams.get("n-head-layers", 1),
            edge_dropout=hyperparams.get("edge-dropout", 0.0),
            sage_aggr=hyperparams.get("sage-aggr", "mean"),
            feature_noise_std=hyperparams.get("feature-noise-std", 0.0),
            add_route_shared_edges=hyperparams.get("add-route-shared-edges", False),
        ).to(device)
    else:
        model = GATEmbedding(
            link_features_in=link_features,
            stop_features_in=stop_features,
            gnn_hidden_dim=hyperparams["gnn-hidden-dim"],
            gnn_output_dim=hyperparams["gnn-output-dim"],
            gnn_layers=hyperparams["gnn-layers"],
            skip_connections=hyperparams.get("skip-connections", True),
            global_skip_connection=hyperparams.get("global-skip-connection", False),
            dropout=hyperparams["dropout"],
            attn_dropout=hyperparams.get("attn-dropout"),
            num_heads=hyperparams["num-heads"],
            head_hidden_dim=hyperparams["head-hidden-dim"],
            head_type=hyperparams.get("head-type", "film"),
            head_time_concat=hyperparams.get("head-time-concat", True),
            n_head_layers=hyperparams.get("n-head-layers", 1),
            edge_dropout=hyperparams.get("edge-dropout", 0.0),
            feature_noise_std=hyperparams.get("feature-noise-std", 0.0),
            add_route_shared_edges=hyperparams.get("add-route-shared-edges", False),
        ).to(device)

    n_params = sum(
        p.numel() for p in model.parameters()
        if not isinstance(p, torch.nn.parameter.UninitializedParameter)
    )
    logger.info("Model: %s (%s parameters)", args.model, f"{n_params:,}")

    # Initialize optimizer — separate LR groups for GNN backbone vs heads
    gnn_lr_factor = hyperparams.get("gnn-lr-factor", 1.0)
    base_lr = hyperparams["learning-rate"]
    gnn_lr = base_lr * gnn_lr_factor
    # Split parameters into GNN backbone vs prediction heads
    head_names = {
        "link_head", "stop_head", "stop_classifier", "embedding_norm",
    }
    gnn_params = []
    head_params = []
    for name, param in model.named_parameters():
        if any(name.startswith(h) for h in head_names):
            head_params.append(param)
        else:
            gnn_params.append(param)
    gnn_wd = hyperparams.get("gnn-weight-decay", hyperparams["weight-decay"])
    head_wd = hyperparams["weight-decay"]
    if gnn_lr_factor != 1.0 or gnn_wd != head_wd:
        logger.info("Separate LR: GNN=%.1e (%.2fx), heads=%.1e | WD: GNN=%.1e, heads=%.1e",
                   gnn_lr, gnn_lr_factor, base_lr, gnn_wd, head_wd)
    optimizer = torch.optim.AdamW(
        [
            {"params": gnn_params, "lr": gnn_lr, "weight_decay": gnn_wd},
            {"params": head_params, "lr": base_lr, "weight_decay": head_wd},
        ],
    )
    # Learning rate scheduler
    scheduler_type = hyperparams.get("scheduler-type", "plateau")
    min_lr = hyperparams.get("min-lr", 1e-6)
    if scheduler_type == "cosine":
        scheduler_T0 = hyperparams.get("scheduler-T0", 15)
        scheduler_T_mult = hyperparams.get("scheduler-T-mult", 2)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_T0,
            T_mult=scheduler_T_mult,
            eta_min=min_lr,
        )
        logger.info("Using CosineAnnealingWarmRestarts (T0=%d, T_mult=%d, min_lr=%.1e)",
                   scheduler_T0, scheduler_T_mult, min_lr)
    elif scheduler_type == "onecycle":
        epochs = hyperparams["epochs"]
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[gnn_lr, base_lr],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=100,
        )
        logger.info("Using OneCycleLR (max_lr=%.1e, steps/epoch=%d, epochs=%d)",
                   hyperparams["learning-rate"], steps_per_epoch, epochs)
    else:
        scheduler_factor = hyperparams.get("scheduler-factor", 0.5)
        scheduler_patience = hyperparams.get("scheduler-patience", 5)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )
        logger.info("Using ReduceLROnPlateau (factor=%.2f, patience=%d, min_lr=%.1e)",
                   scheduler_factor, scheduler_patience, min_lr)

    # Training loop
    epochs = hyperparams["epochs"]
    early_stopping_patience = hyperparams["early-stopping-patience"]
    early_stopping_min_delta = hyperparams["early-stopping-min-delta"]
    use_amp = hyperparams.get("use-amp", False) and device.type == "cuda"
    link_reg_weight = hyperparams.get("link-reg-weight", 1.0)
    stop_reg_weight = hyperparams.get("stop-reg-weight", 1.0)
    stop_cls_weight = hyperparams.get("stop-cls-weight", 1.0)
    loss_type = hyperparams.get("loss-type", "mse")
    huber_delta = hyperparams.get("huber-delta", 1.0)

    scaler = GradScaler("cuda") if use_amp else None

    logger.info(
        "Training for %d epochs (early_stop=%d, lr=%.1e, loss=%s%s)",
        epochs, early_stopping_patience, hyperparams["learning-rate"],
        loss_type, f" delta={huber_delta}" if loss_type == "huber" else "",
    )

    best_val_loss = float('inf')
    best_model_state = None
    best_val_metrics = None
    best_oversmoothing_metrics = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # Train on train graph
        train_loss, train_link_loss, train_stop_reg_loss, train_stop_cls_loss, train_stop_loss = train_epoch(
            model, train_graph, train_loader, optimizer, device,
            link_reg_weight=link_reg_weight,
            stop_reg_weight=stop_reg_weight,
            stop_cls_weight=stop_cls_weight,
            grad_clip=hyperparams["grad-clip"],
            scaler=scaler,
            loss_type=loss_type,
            huber_delta=huber_delta,
            focal_gamma=hyperparams.get("focal-gamma", 0.0),
            scheduler=scheduler if isinstance(scheduler, (CosineAnnealingWarmRestarts, OneCycleLR)) else None,
        )

        if epoch % LOG_INTERVAL == 0 or epoch == 1:
            # Evaluate on val graph (different regions, different graph!)
            if val_loader is not None and val_graph is not None:
                val_loss, val_link_loss, val_stop_reg_loss, val_stop_cls_loss, val_stop_loss, val_metrics = evaluate(
                    model, val_graph, val_loader, device,
                    link_reg_weight=link_reg_weight,
                    stop_reg_weight=stop_reg_weight,
                    stop_cls_weight=stop_cls_weight,
                    use_amp=use_amp,
                    link_scaler=train_dataset.link_scaler,
                    stop_scaler=train_dataset.stop_scaler,
                    log_transform=log_transform,
                    link_freeflow_refs=val_dataset.link_freeflow_refs if val_dataset else None,
                    freeflow_ratio=freeflow_ratio,
                    link_municipalities=val_dataset.link_municipalities if val_dataset else None,
                    stop_municipalities=val_dataset.stop_municipalities if val_dataset else None,
                )
            else:
                # Fall back to train loss if no val data
                val_loss = train_loss
                val_link_loss = train_link_loss
                val_stop_reg_loss = train_stop_reg_loss
                val_stop_cls_loss = train_stop_cls_loss
                val_stop_loss = train_stop_loss
                val_metrics = {}

            link_mae = val_metrics.get("val/link/mae", 0)
            link_rmse = val_metrics.get("val/link/rmse", 0)
            link_bias = val_metrics.get("val/link/bias", 0)
            link_smape = val_metrics.get("val/link/smape", 0)
            link_node_r2 = val_metrics.get("val/link/node_r2", 0)
            link_node_mape = val_metrics.get("val/link/node_mape", 0)
            link_node_mae = val_metrics.get("val/link/node_mae", 0)
            stop_mae = val_metrics.get("val/stop/mae", 0)
            stop_rmse = val_metrics.get("val/stop/rmse", 0)
            stop_smape = val_metrics.get("val/stop/smape", 0)
            stop_node_r2 = val_metrics.get("val/stop/node_r2", 0)
            stop_node_mape = val_metrics.get("val/stop/node_mape", 0)
            stop_accuracy = val_metrics.get("val/stop_accuracy", 0)
            stop_f1 = val_metrics.get("val/stop_f1", 0)
            stop_auc = val_metrics.get("val/stop_auc", 0)

            # Seasonal MASE / RMSSE for val
            link_smase_str, link_srmsse_str = "N/A", "N/A"
            if link_mae > 0:
                link_smase_str, link_srmsse_str = _compute_smase_metrics(
                    df_travel_val, df_travel_val_full, "link",
                    link_mae, link_rmse, val_metrics, "val/link",
                )

            stop_smase_str, stop_srmsse_str = "N/A", "N/A"
            if stop_mae > 0:
                stop_smase_str, stop_srmsse_str = _compute_smase_metrics(
                    df_dwell_val[df_dwell_val["travel_time"] > 0],
                    df_dwell_val_full[df_dwell_val_full["travel_time"] > 0],
                    "stop", stop_mae, stop_rmse, val_metrics, "val/stop",
                )

            logger.info(
                "Epoch %03d | train=%.4f (link=%.4f, stop=%.4f [reg=%.4f, cls=%.4f]) | val=%.4f (link=%.4f, stop=%.4f [reg=%.4f, cls=%.4f])",
                epoch, train_loss, train_link_loss, train_stop_loss, train_stop_reg_loss, train_stop_cls_loss,
                val_loss, val_link_loss, val_stop_loss, val_stop_reg_loss, val_stop_cls_loss
            )

            # Compute oversmoothing metrics on train graph
            oversmoothing_metrics = compute_oversmoothing_metrics(model, train_graph, device)

            logger.info(
                "  Link  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s | MAD: %6.3f | MADGap: %6.3f",
                link_mae, link_rmse, link_smape,
                link_smase_str, link_srmsse_str,
                oversmoothing_metrics.get("link/mad", 0), oversmoothing_metrics.get("link/mad_gap", 0),
            )
            logger.info(
                "  Stop  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s | F1: %6.3f | AUC: %6.3f | Acc: %6.3f | MAD: %6.3f | MADGap: %6.3f",
                stop_mae, stop_rmse, stop_smape,
                stop_smase_str, stop_srmsse_str,
                stop_f1, stop_auc, stop_accuracy,
                oversmoothing_metrics.get("stop/mad", 0), oversmoothing_metrics.get("stop/mad_gap", 0),
            )

            if run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                log_dict = {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train/loss": train_loss,
                    "train_link/loss": train_link_loss,
                    "train_stop/loss": train_stop_loss,
                    "train_stop/reg_loss": train_stop_reg_loss,
                    "train_stop/cls_loss": train_stop_cls_loss,
                    "val_transfer/loss": val_loss,
                    "val_transfer_link/loss": val_link_loss,
                    "val_transfer_stop/loss": val_stop_loss,
                    "val_transfer_stop/reg_loss": val_stop_reg_loss,
                    "val_transfer_stop/cls_loss": val_stop_cls_loss,
                }

                # Validation metrics (transfer/OOD regions)
                log_dict.update(split_metrics_by_node_type(val_metrics, "val_transfer"))

                # Combined metrics for joint link+stop optimization in sweeps (geometric mean)
                link_srmsse = log_dict.get("val_transfer_link/srmsse")
                stop_srmsse = log_dict.get("val_transfer_stop/srmsse")
                if link_srmsse is not None and stop_srmsse is not None:
                    log_dict["val_transfer_combined/srmsse"] = (link_srmsse * stop_srmsse) ** 0.5
                link_smase_tr = log_dict.get("val_transfer_link/smase")
                stop_smase_tr = log_dict.get("val_transfer_stop/smase")
                if link_smase_tr is not None and stop_smase_tr is not None:
                    log_dict["val_transfer_combined/smase"] = (link_smase_tr * stop_smase_tr) ** 0.5

                # Oversmoothing metrics
                log_dict.update(split_oversmoothing_metrics(oversmoothing_metrics))

                # Same-region temporal evaluation (train regions × val time window)
                if train_valtime_loader is not None:
                    _, _, _, _, _, train_valtime_metrics = evaluate(
                        model, train_graph, train_valtime_loader, device,
                        link_reg_weight=link_reg_weight,
                        stop_reg_weight=stop_reg_weight,
                        stop_cls_weight=stop_cls_weight,
                        use_amp=use_amp,
                        link_scaler=train_dataset.link_scaler,
                        stop_scaler=train_dataset.stop_scaler,
                        prefix="train_region_val",
                        log_transform=log_transform,
                        link_freeflow_refs=train_valtime_dataset.link_freeflow_refs if train_valtime_dataset else None,
                        freeflow_ratio=freeflow_ratio,
                        link_municipalities=train_valtime_dataset.link_municipalities if train_valtime_dataset else None,
                        stop_municipalities=train_valtime_dataset.stop_municipalities if train_valtime_dataset else None,
                    )
                    # Original (in-distribution) validation metrics
                    log_dict.update(split_metrics_by_node_type(train_valtime_metrics, "val_original"))

                run.log(log_dict)

            # Early stopping — geometric mean of link_sRMSSE and stop_sRMSSE.
            # Val loss (Huber on normalised targets) diverges from sRMSSE immediately;
            # using it overshoots the best checkpoint by patience epochs.
            # Geometric mean matches the sweep objective and balances relative
            # improvements across both heads when they sit on different scales.
            link_srmsse_val = val_metrics.get("val/link/srmsse")
            stop_srmsse_val = val_metrics.get("val/stop/srmsse")
            if link_srmsse_val is not None and stop_srmsse_val is not None:
                early_stop_metric = (link_srmsse_val * stop_srmsse_val) ** 0.5
            else:
                early_stop_metric = val_loss  # fallback before sRMSSE is available

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(early_stop_metric)

            if early_stop_metric < best_val_loss - early_stopping_min_delta:
                best_val_loss = early_stop_metric
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_val_metrics = dict(val_metrics)
                best_oversmoothing_metrics = dict(oversmoothing_metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss, best_val_metrics, best_oversmoothing_metrics


def main() -> None:
    """Entry point for embedding training."""
    args = parse_args()
    setup_logging(level=args.log_level)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load hyperparameters for selected model
    if args.model == "graphsage":
        hyperparams = EMBEDDING_GRAPHSAGE_DEFAULTS.copy()
    elif args.model == "gat":
        hyperparams = EMBEDDING_GAT_DEFAULTS.copy()
    else:  # xgboost
        hyperparams = EMBEDDING_XGBOOST_DEFAULTS.copy()

    # Apply feature overrides from ablation study (via environment variable)
    ablation_overrides = os.environ.get("ABLATION_FEATURE_OVERRIDES")
    if ablation_overrides:
        import json as _json
        overrides = _json.loads(ablation_overrides)
        for key, value in overrides.items():
            if key not in hyperparams:
                raise ValueError(f"Unknown ablation override key: {key}")
            logger.info("Ablation override: %s = %s (was %s)", key, value, hyperparams[key])
            hyperparams[key] = value

    logger.info("Model: %s", args.model)

    logger.info("Loading preprocessed parquet files...")
    df_travel_all = pd.read_parquet(args.travel_times_parquet, engine="pyarrow")
    df_dwell_all = pd.read_parquet(args.dwell_times_parquet, engine="pyarrow")

    # Ensure datetime columns are timezone-aware
    for col in ["from_time", "to_time"]:
        if df_travel_all[col].dt.tz is None:
            df_travel_all[col] = df_travel_all[col].dt.tz_localize("UTC")
        if df_dwell_all[col].dt.tz is None:
            df_dwell_all[col] = df_dwell_all[col].dt.tz_localize("UTC")

    logger.info(
        "Loaded parquet: %d travel time records, %d dwell time records",
        len(df_travel_all), len(df_dwell_all),
    )

    # Route filtering (optional): keep only data from one municipality+route
    if args.route_municipality is not None:
        df_travel_all = df_travel_all[
            df_travel_all["municipality"] == args.route_municipality
        ]
        df_dwell_all = df_dwell_all[
            df_dwell_all["municipality"] == args.route_municipality
        ]
        if args.route_id is not None:
            route_filter_val = int(args.route_id) if args.route_id.isdigit() else args.route_id
            if "route" in df_travel_all.columns:
                df_travel_all = df_travel_all[df_travel_all["route"] == route_filter_val]
            if "route" in df_dwell_all.columns:
                df_dwell_all = df_dwell_all[df_dwell_all["route"] == route_filter_val]
        logger.info(
            "After route filter (%s:%s): %d travel, %d dwell",
            args.route_municipality, args.route_id,
            len(df_travel_all), len(df_dwell_all),
        )

    # Discover municipalities present in the parquet data
    parquet_municipalities = (
        set(df_travel_all["municipality"].unique()) &
        set(df_dwell_all["municipality"].unique())
    )

    # Discover available municipalities (must have both artifacts and parquet data)
    available_regions = discover_regions(args.artifacts_dir, parquet_municipalities)

    # Assign regions to train/val/test (stratified by category)
    train_regions, val_regions, test_regions = assign_regions_to_splits(
        available_regions
    )

    logger.info("Region assignment (predefined in config):")
    logger.info("  Train (%d): %s", len(train_regions), train_regions)
    logger.info("  Val   (%d): %s", len(val_regions), val_regions)
    logger.info("  Test  (%d): %s", len(test_regions), test_regions)

    # Log category breakdown
    for split_name, regions in [("Train", train_regions), ("Val", val_regions), ("Test", test_regions)]:
        cat_counts: Dict[str, int] = {}
        for r in regions:
            cat = MUNICIPALITY_TO_CATEGORY[r]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        logger.info("  %s categories: %s", split_name, dict(sorted(cat_counts.items())))

    # Build artifact file paths for each split's regions
    train_link_csvs, train_stop_csvs, train_edge_csvs = \
        get_region_artifact_paths(train_regions, args.artifacts_dir)
    val_link_csvs, val_stop_csvs, val_edge_csvs = \
        get_region_artifact_paths(val_regions, args.artifacts_dir)
    test_link_csvs, test_stop_csvs, test_edge_csvs = \
        get_region_artifact_paths(test_regions, args.artifacts_dir)

    # Split parquet data by municipality assignment
    df_travel_train_full = df_travel_all[df_travel_all["municipality"].isin(train_regions)]
    df_dwell_train_full = df_dwell_all[df_dwell_all["municipality"].isin(train_regions)]

    df_travel_val_full = df_travel_all[df_travel_all["municipality"].isin(val_regions)]
    df_dwell_val_full = df_dwell_all[df_dwell_all["municipality"].isin(val_regions)]

    df_travel_test_full = df_travel_all[df_travel_all["municipality"].isin(test_regions)]
    df_dwell_test_full = df_dwell_all[df_dwell_all["municipality"].isin(test_regions)]

    # Free the combined DataFrames
    del df_travel_all, df_dwell_all

    logger.info(
        "Split data: train=%d travel + %d dwell, val=%d travel + %d dwell, test=%d travel + %d dwell",
        len(df_travel_train_full), len(df_dwell_train_full),
        len(df_travel_val_full), len(df_dwell_val_full),
        len(df_travel_test_full), len(df_dwell_test_full),
    )

    # Build separate hetero graphs for train/val/test regions
    logger.info("Building heterogeneous graphs for each region group...")

    train_graph, train_link_nodes, train_stop_nodes, train_norm_stats = build_hetero_graph(
        train_link_csvs, train_stop_csvs, train_edge_csvs,
        feature_config=hyperparams,
    )
    logger.info(
        "Train graph: %d link nodes, %d stop nodes",
        len(train_link_nodes), len(train_stop_nodes),
    )

    # Val/test graphs use training normalization statistics for feature consistency
    val_graph: Optional[HeteroData] = None
    val_link_nodes: Optional[pd.DataFrame] = None
    val_stop_nodes: Optional[pd.DataFrame] = None
    if len(df_travel_val_full) > 0:
        val_graph, val_link_nodes, val_stop_nodes, _ = build_hetero_graph(
            val_link_csvs, val_stop_csvs, val_edge_csvs,
            feature_config=hyperparams,
            feature_norm_stats=train_norm_stats,
        )
        logger.info(
            "Val graph: %d link nodes, %d stop nodes",
            len(val_link_nodes), len(val_stop_nodes),
        )

    test_graph: Optional[HeteroData] = None
    test_link_nodes: Optional[pd.DataFrame] = None
    test_stop_nodes: Optional[pd.DataFrame] = None
    if len(df_travel_test_full) > 0:
        test_graph, test_link_nodes, test_stop_nodes, _ = build_hetero_graph(
            test_link_csvs, test_stop_csvs, test_edge_csvs,
            feature_config=hyperparams,
            feature_norm_stats=train_norm_stats,
        )
        logger.info(
            "Test graph: %d link nodes, %d stop nodes",
            len(test_link_nodes), len(test_stop_nodes),
        )

    # Determine time range across ALL data for consistent split boundaries
    overall_start, overall_end = get_time_range(
        pd.concat([df_travel_train_full, df_travel_val_full, df_travel_test_full], ignore_index=True),
        pd.concat([df_dwell_train_full, df_dwell_val_full, df_dwell_test_full], ignore_index=True),
    )
    logger.info("Time range: %s to %s", overall_start, overall_end)

    split_boundaries = generate_split_boundaries(
        start_time=overall_start,
        end_time=overall_end,
        val_size=0.5,
        n_splits=N_SPLITS,
    )

    # Create wandb group name
    model_name = EMBEDDING_MODEL_DISPLAY_NAMES.get(args.model, args.model)
    group_name = f"{model_name}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}"

    # Determine splits to train
    if args.sweep:
        splits_to_train = [N_SPLITS - 1]
        logger.info("[SWEEP MODE] Training on split %d only", N_SPLITS - 1)
    else:
        splits_to_train = list(range(N_SPLITS))

    # Store test metrics for aggregation
    all_test_metrics: List[Dict[str, float]] = []
    all_val_losses: List[float] = []

    for split_idx in splits_to_train:
        # Reset seed per fold so results are identical regardless of which
        # folds are trained (--sweep vs full run).
        seed_all(args.seed + split_idx)
        logger.info("=" * 60)
        logger.info("SPLIT %d/%d", split_idx + 1, N_SPLITS)
        logger.info("=" * 60)

        # Get split boundaries
        train_start, train_end, val_end, test_end = split_boundaries[split_idx]
        val_start = train_end  # validation starts where train ends
        test_start = val_end   # test starts where validation ends
        logger.info("Train: %s to %s", train_start, train_end)
        logger.info("Val:   %s to %s", val_start, val_end)
        logger.info("Test:  %s to %s", test_start, test_end)

        # Initialize wandb
        if not args.no_wandb:
            wandb_config = {
                "model_type": args.model,
                "model_name": model_name,
                "stage": 1,
                "DS_n_splits": N_SPLITS,
                "seed": args.seed,
                "split_idx": split_idx,
                "train_regions": train_regions,
                "val_regions": val_regions,
                "test_regions": test_regions,
            }
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=group_name,
                name=f"{split_idx}_{group_name}",
                config=wandb_config,
                reinit=True,
            )

            # Define metric hierarchies for wandb dashboard
            wandb.define_metric("epoch")
            for prefix in [
                "train/*", "train_link/*", "train_stop/*",
                "val_transfer/*", "val_transfer_link/*", "val_transfer_stop/*",
                "val_transfer_combined/*",
                "val_original_link/*", "val_original_stop/*",
                "oversmoothing_link/*", "oversmoothing_stop/*",
                "best_val_transfer_link/*", "best_val_transfer_stop/*",
                "best_val_transfer_combined/*",
                "best_test_transfer_link/*", "best_test_transfer_stop/*",
                "best_oversmoothing_link/*", "best_oversmoothing_stop/*",
            ]:
                wandb.define_metric(prefix, step_metric="epoch")

            # Summary statistics: min for losses/errors, max for classification, last for oversmoothing
            for metric in [
                "train/loss", "val_transfer/loss",
                "train_link/loss", "train_stop/loss", "train_stop/reg_loss", "train_stop/cls_loss",
                "val_transfer_link/loss", "val_transfer_stop/loss",
                "val_transfer_stop/reg_loss", "val_transfer_stop/cls_loss",
                "val_transfer_link/mae", "val_transfer_link/rmse",
                "val_transfer_link/smase", "val_transfer_link/srmsse",
                "val_transfer_stop/mae", "val_transfer_stop/rmse",
                "val_transfer_stop/smase", "val_transfer_stop/srmsse",
                "val_transfer_combined/srmsse",
                "val_original_link/mae", "val_original_link/rmse",
                "val_original_link/smase", "val_original_link/srmsse",
                "val_original_stop/mae", "val_original_stop/rmse",
                "val_original_stop/smase", "val_original_stop/srmsse",
                "val_transfer_stop/brier", "val_original_stop/brier",
                "val_transfer_link/node_mape", "val_transfer_link/node_mae", "val_transfer_link/node_smape",
                "val_transfer_stop/node_mape", "val_transfer_stop/node_mae", "val_transfer_stop/node_smape",
                "val_original_link/node_mape", "val_original_link/node_mae", "val_original_link/node_smape",
                "val_original_stop/node_mape", "val_original_stop/node_mae", "val_original_stop/node_smape",
            ]:
                wandb.define_metric(metric, summary="min")
            for metric in [
                "val_transfer_stop/accuracy", "val_transfer_stop/precision",
                "val_transfer_stop/recall", "val_transfer_stop/f1",
                "val_transfer_stop/auc",
                "val_transfer_link/node_r2", "val_transfer_stop/node_r2",
                "val_original_link/node_r2", "val_original_stop/node_r2",
                "val_original_stop/accuracy", "val_original_stop/precision",
                "val_original_stop/recall", "val_original_stop/f1",
                "val_original_stop/auc",
            ]:
                wandb.define_metric(metric, summary="max")
            for metric in [
                "oversmoothing_link/mad", "oversmoothing_link/mad_gap",
                "oversmoothing_stop/mad", "oversmoothing_stop/mad_gap",
            ]:
                wandb.define_metric(metric, summary="last")

            # Update hyperparams from sweep config
            sweep_config = dict(wandb.config)
            for key in hyperparams.keys():
                if key in sweep_config:
                    sweep_value = sweep_config[key]
                    if sweep_value != hyperparams[key]:
                        logger.info("Sweep override: %s = %s (was %s)", key, sweep_value, hyperparams[key])
                        hyperparams[key] = sweep_value

            run.config.update(hyperparams)
        else:
            run = None

        # Filter data by time window
        df_travel_train = df_travel_train_full[
            (df_travel_train_full["from_time"] >= train_start) &
            (df_travel_train_full["from_time"] < train_end)
        ]
        df_dwell_train = df_dwell_train_full[
            (df_dwell_train_full["from_time"] >= train_start) &
            (df_dwell_train_full["from_time"] < train_end)
        ]

        # Val: val regions × val time window
        df_travel_val = df_travel_val_full[
            (df_travel_val_full["from_time"] >= val_start) &
            (df_travel_val_full["from_time"] < val_end)
        ] if len(df_travel_val_full) > 0 else df_travel_val_full

        df_dwell_val = df_dwell_val_full[
            (df_dwell_val_full["from_time"] >= val_start) &
            (df_dwell_val_full["from_time"] < val_end)
        ] if len(df_dwell_val_full) > 0 else df_dwell_val_full

        # Test: test regions × test time window
        df_travel_test = df_travel_test_full[
            (df_travel_test_full["from_time"] >= test_start) &
            (df_travel_test_full["from_time"] < test_end)
        ] if len(df_travel_test_full) > 0 else df_travel_test_full

        df_dwell_test = df_dwell_test_full[
            (df_dwell_test_full["from_time"] >= test_start) &
            (df_dwell_test_full["from_time"] < test_end)
        ] if len(df_dwell_test_full) > 0 else df_dwell_test_full

        # Same-region temporal evaluation
        df_travel_train_valtime = df_travel_train_full[
            (df_travel_train_full["from_time"] >= val_start) &
            (df_travel_train_full["from_time"] < val_end)
        ]
        df_dwell_train_valtime = df_dwell_train_full[
            (df_dwell_train_full["from_time"] >= val_start) &
            (df_dwell_train_full["from_time"] < val_end)
        ]

        df_travel_train_testtime = df_travel_train_full[
            (df_travel_train_full["from_time"] >= test_start) &
            (df_travel_train_full["from_time"] < test_end)
        ]
        df_dwell_train_testtime = df_dwell_train_full[
            (df_dwell_train_full["from_time"] >= test_start) &
            (df_dwell_train_full["from_time"] < test_end)
        ]

        logger.info(
            "Split %d data: travel (train=%d, val=%d, test=%d), dwell (train=%d, val=%d, test=%d)",
            split_idx,
            len(df_travel_train), len(df_travel_val), len(df_travel_test),
            len(df_dwell_train), len(df_dwell_val), len(df_dwell_test),
        )
        logger.info(
            "  Same-region temporal: travel (valtime=%d, testtime=%d), dwell (valtime=%d, testtime=%d)",
            len(df_travel_train_valtime), len(df_travel_train_testtime),
            len(df_dwell_train_valtime), len(df_dwell_train_testtime),
        )

        # Read log-transform from hyperparams
        log_transform = hyperparams.get("log-transform", False)
        if log_transform:
            logger.info("Log transform enabled: travel/dwell times will be log1p-transformed")

        # Read residual modes from hyperparams
        travel_time_residual_mode, _dwell_time_residual_mode = get_residual_modes(hyperparams)
        logger.info("Residual modes: travel_time=%s, dwell_time=%s", travel_time_residual_mode, _dwell_time_residual_mode)

        # Build freeflow_times dict for freeflow residual/ratio mode
        # Separate dicts for train/val/test since they have different link nodes
        freeflow_times_train: dict[str, float] | None = None
        freeflow_times_val: dict[str, float] | None = None
        freeflow_times_test: dict[str, float] | None = None
        freeflow_ratio = travel_time_residual_mode == "freeflow_ratio"

        if travel_time_residual_mode in ("freeflow", "freeflow_ratio"):
            if "osm_travel_time" not in train_link_nodes.columns:
                raise ValueError(
                    f"travel-time-residual-mode='{travel_time_residual_mode}' requires "
                    "'osm_travel_time' column in link_nodes CSV. "
                    "Run build_graph with --use-osm-travel-time."
                )
            freeflow_times_train = dict(zip(
                train_link_nodes["id"].astype(str),
                train_link_nodes["osm_travel_time"].astype(float),
            ))
            logger.info("Freeflow %s mode: using osm_travel_time from %d train links",
                       "ratio" if freeflow_ratio else "residual", len(freeflow_times_train))

            if val_link_nodes is not None and "osm_travel_time" in val_link_nodes.columns:
                freeflow_times_val = dict(zip(
                    val_link_nodes["id"].astype(str),
                    val_link_nodes["osm_travel_time"].astype(float),
                ))
            if test_link_nodes is not None and "osm_travel_time" in test_link_nodes.columns:
                freeflow_times_test = dict(zip(
                    test_link_nodes["id"].astype(str),
                    test_link_nodes["osm_travel_time"].astype(float),
                ))
        elif travel_time_residual_mode != "none":
            raise ValueError(
                f"Unsupported travel-time-residual-mode='{travel_time_residual_mode}'. "
                f"Only 'freeflow', 'freeflow_ratio', and 'none' are supported "
                f"(no historical averages are available here)."
            )

        # Subsample applied per-epoch in the train loader (fresh subset each iter),
        # not upfront — that way scalers are fit on the full distribution and the
        # model sees more diverse data over the run.
        train_subsample = hyperparams.get("train-subsample", 1.0)
        if train_subsample < 1.0:
            logger.info(
                "Train subsample: %.0f%% per epoch (resampled each iteration)",
                train_subsample * 100,
            )

        # Train dataset fits the scalers (using train regions + train time window).
        train_dataset = RawObservationProxyDataset(
            df_travel=df_travel_train,
            df_dwell=df_dwell_train,
            link_nodes=train_link_nodes,
            stop_nodes=train_stop_nodes,
            link_scaler=None,  # Will fit on training data
            stop_scaler=None,  # Will fit on stopped observations only
            log_transform=log_transform,
            freeflow_times=freeflow_times_train,
            freeflow_ratio=freeflow_ratio,
        )

        # Val/test datasets use train scalers (no data leakage)
        # Val uses val graph's nodes, test uses test graph's nodes
        has_val = val_link_nodes is not None and len(df_travel_val) > 0
        has_test = test_link_nodes is not None and len(df_travel_test) > 0

        val_dataset: Optional[RawObservationProxyDataset] = None
        if has_val:
            assert val_link_nodes is not None and val_stop_nodes is not None
            val_dataset = RawObservationProxyDataset(
                df_travel=df_travel_val,
                df_dwell=df_dwell_val,
                link_nodes=val_link_nodes,
                stop_nodes=val_stop_nodes,
                link_scaler=train_dataset.link_scaler,
                stop_scaler=train_dataset.stop_scaler,
                log_transform=log_transform,
                freeflow_times=freeflow_times_val,
                freeflow_ratio=freeflow_ratio,
            )

        test_dataset: Optional[RawObservationProxyDataset] = None
        if has_test:
            assert test_link_nodes is not None and test_stop_nodes is not None
            test_dataset = RawObservationProxyDataset(
                df_travel=df_travel_test,
                df_dwell=df_dwell_test,
                link_nodes=test_link_nodes,
                stop_nodes=test_stop_nodes,
                link_scaler=train_dataset.link_scaler,
                stop_scaler=train_dataset.stop_scaler,
                log_transform=log_transform,
                freeflow_times=freeflow_times_test,
                freeflow_ratio=freeflow_ratio,
            )

        # Same-region temporal datasets (train graph nodes, val/test time windows)
        has_train_valtime = len(df_travel_train_valtime) > 0
        has_train_testtime = len(df_travel_train_testtime) > 0

        train_valtime_dataset: Optional[RawObservationProxyDataset] = None
        if has_train_valtime:
            train_valtime_dataset = RawObservationProxyDataset(
                df_travel=df_travel_train_valtime,
                df_dwell=df_dwell_train_valtime,
                link_nodes=train_link_nodes,
                stop_nodes=train_stop_nodes,
                link_scaler=train_dataset.link_scaler,
                stop_scaler=train_dataset.stop_scaler,
                log_transform=log_transform,
                freeflow_times=freeflow_times_train,
                freeflow_ratio=freeflow_ratio,
            )

        train_testtime_dataset: Optional[RawObservationProxyDataset] = None
        if has_train_testtime:
            train_testtime_dataset = RawObservationProxyDataset(
                df_travel=df_travel_train_testtime,
                df_dwell=df_dwell_train_testtime,
                link_nodes=train_link_nodes,
                stop_nodes=train_stop_nodes,
                link_scaler=train_dataset.link_scaler,
                stop_scaler=train_dataset.stop_scaler,
                log_transform=log_transform,
                freeflow_times=freeflow_times_train,
                freeflow_ratio=freeflow_ratio,
            )

        logger.info(
            "Dataset sizes: train=%d, val=%s, test=%s, train_valtime=%s, train_testtime=%s",
            len(train_dataset),
            len(val_dataset) if val_dataset else "N/A",
            len(test_dataset) if test_dataset else "N/A",
            len(train_valtime_dataset) if train_valtime_dataset else "N/A",
            len(train_testtime_dataset) if train_testtime_dataset else "N/A",
        )

        # Create data loaders
        total_samples = len(train_dataset)
        num_workers = hyperparams.get("num-workers", 8)
        if total_samples < 100000:
            num_workers = 0
            logger.info("Using single-threaded data loading (dataset size=%d)", total_samples)
        elif device.type != "cuda":
            num_workers = 0
            logger.info("Using single-threaded data loading (CPU mode)")
        batch_size = hyperparams["batch-size"]

        train_loader = _make_loader(
            train_dataset, batch_size, num_workers, device,
            shuffle=True, subsample=train_subsample,
        )

        val_loader: Optional[ProxyTaskBatchIterator] = None
        if val_dataset is not None:
            val_loader = _make_loader(val_dataset, batch_size, num_workers, device)

        test_loader: Optional[ProxyTaskBatchIterator] = None
        if test_dataset is not None:
            test_loader = _make_loader(test_dataset, batch_size, num_workers, device)

        # Same-region temporal loaders (train graph, val/test time windows)
        train_valtime_loader: Optional[ProxyTaskBatchIterator] = None
        if train_valtime_dataset is not None:
            train_valtime_loader = _make_loader(train_valtime_dataset, batch_size, num_workers, device)

        train_testtime_loader: Optional[ProxyTaskBatchIterator] = None
        if train_testtime_dataset is not None:
            train_testtime_loader = _make_loader(train_testtime_dataset, batch_size, num_workers, device)

        # Hyperparams referenced by the shared post-training eval path.  These
        # used to live inside the inline GNN training block; lifting them here
        # so both the GNN and XGBoost dispatch paths can hand them to evaluate().
        link_reg_weight = hyperparams.get("link-reg-weight", 1.0)
        stop_reg_weight = hyperparams.get("stop-reg-weight", 1.0)
        stop_cls_weight = hyperparams.get("stop-cls-weight", 1.0)
        use_amp = hyperparams.get("use-amp", False) and device.type == "cuda"

        # Initialize and train the model for this fold.  GNN path runs the
        # full epoch loop; XGBoost path is one-shot fit + single val eval.
        if args.model == "xgboost":
            model, best_val_loss, best_val_metrics, best_oversmoothing_metrics = _train_xgboost_split(
                args, hyperparams, device,
                train_graph, val_graph,
                train_dataset, val_dataset, val_loader,
                df_travel_val, df_travel_val_full, df_dwell_val, df_dwell_val_full,
                log_transform, freeflow_ratio, run,
            )
        else:
            model, best_val_loss, best_val_metrics, best_oversmoothing_metrics = _train_gnn_split(
                args, hyperparams, device,
                train_graph, val_graph, train_regions,
                train_dataset, val_dataset, train_valtime_dataset,
                train_loader, val_loader, train_valtime_loader,
                df_travel_val, df_travel_val_full, df_dwell_val, df_dwell_val_full,
                log_transform, freeflow_ratio, run,
            )


        logger.info("Split %d complete - best early_stop_metric (sMASE): %.6f", split_idx + 1, best_val_loss)
        all_val_losses.append(best_val_loss)

        # Train evaluation (best model on train regions × train time window)
        logger.info("Evaluating on train set (in-distribution)")
        train_eval_loader = _make_loader(train_dataset, batch_size, num_workers, device)
        (
            train_eval_loss, train_eval_link_loss,
            train_eval_stop_reg_loss, train_eval_stop_cls_loss,
            train_eval_stop_loss, train_eval_metrics
        ) = evaluate(
            model, train_graph, train_eval_loader, device,
            link_reg_weight=link_reg_weight,
            stop_reg_weight=stop_reg_weight,
            stop_cls_weight=stop_cls_weight,
            use_amp=use_amp,
            link_scaler=train_dataset.link_scaler,
            stop_scaler=train_dataset.stop_scaler,
            prefix="train_eval",
            log_transform=log_transform,
            link_freeflow_refs=train_dataset.link_freeflow_refs,
            freeflow_ratio=freeflow_ratio,
            link_municipalities=train_dataset.link_municipalities,
            stop_municipalities=train_dataset.stop_municipalities,
        )

        tr_eval_link_mae = train_eval_metrics.get("train_eval/link/mae", 0)
        tr_eval_link_rmse = train_eval_metrics.get("train_eval/link/rmse", 0)
        tr_eval_link_smape = train_eval_metrics.get("train_eval/link/smape", 0)
        tr_eval_stop_mae = train_eval_metrics.get("train_eval/stop/mae", 0)
        tr_eval_stop_rmse = train_eval_metrics.get("train_eval/stop/rmse", 0)
        tr_eval_stop_smape = train_eval_metrics.get("train_eval/stop/smape", 0)
        tr_eval_stop_f1 = train_eval_metrics.get("train_eval/stop_f1", 0)
        tr_eval_stop_auc = train_eval_metrics.get("train_eval/stop_auc", 0)
        tr_eval_stop_accuracy = train_eval_metrics.get("train_eval/stop_accuracy", 0)

        tr_eval_link_smase_str, tr_eval_link_srmsse_str = "N/A", "N/A"
        if tr_eval_link_mae > 0:
            _dummy_tr: Dict[str, Any] = {}
            tr_eval_link_smase_str, tr_eval_link_srmsse_str = _compute_smase_metrics(
                df_travel_train, df_travel_train_full, "link",
                tr_eval_link_mae, tr_eval_link_rmse, _dummy_tr, "train/link",
            )
            train_eval_metrics.update(_dummy_tr)

        tr_eval_stop_smase_str, tr_eval_stop_srmsse_str = "N/A", "N/A"
        if tr_eval_stop_mae > 0:
            _dummy_tr2: Dict[str, Any] = {}
            tr_eval_stop_smase_str, tr_eval_stop_srmsse_str = _compute_smase_metrics(
                df_dwell_train[df_dwell_train["travel_time"] > 0],
                df_dwell_train_full[df_dwell_train_full["travel_time"] > 0],
                "stop", tr_eval_stop_mae, tr_eval_stop_rmse, _dummy_tr2, "train/stop",
            )
            train_eval_metrics.update(_dummy_tr2)

        logger.info(
            "Train eval | loss=%.4f (link=%.4f, stop=%.4f)",
            train_eval_loss, train_eval_link_loss, train_eval_stop_loss,
        )
        logger.info(
            "  Link  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s",
            tr_eval_link_mae, tr_eval_link_rmse, tr_eval_link_smape,
            tr_eval_link_smase_str, tr_eval_link_srmsse_str,
        )
        logger.info(
            "  Stop  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s | F1: %6.3f | AUC: %6.3f | Acc: %6.3f",
            tr_eval_stop_mae, tr_eval_stop_rmse, tr_eval_stop_smape,
            tr_eval_stop_smase_str, tr_eval_stop_srmsse_str,
            tr_eval_stop_f1, tr_eval_stop_auc, tr_eval_stop_accuracy,
        )

        if run is not None:
            run.summary.update(split_metrics_by_node_type(train_eval_metrics, "train", best_prefix=True))

        del train_eval_loader

        # Test evaluation (on test graph + test time window)
        if test_loader is not None and test_graph is not None and test_dataset is not None:
            logger.info("Evaluating on test set (different regions)")

            test_loss, test_link_loss, test_stop_reg_loss, test_stop_cls_loss, test_stop_loss, test_metrics = evaluate(
                model, test_graph, test_loader, device,
                link_reg_weight=link_reg_weight,
                stop_reg_weight=stop_reg_weight,
                stop_cls_weight=stop_cls_weight,
                use_amp=use_amp,
                link_scaler=train_dataset.link_scaler,
                stop_scaler=train_dataset.stop_scaler,
                prefix="test",
                log_transform=log_transform,
                link_freeflow_refs=test_dataset.link_freeflow_refs,
                freeflow_ratio=freeflow_ratio,
                link_municipalities=test_dataset.link_municipalities,
                stop_municipalities=test_dataset.stop_municipalities,
            )

            test_link_mae = test_metrics.get("test/link/mae", 0)
            test_link_rmse = test_metrics.get("test/link/rmse", 0)
            test_link_smape = test_metrics.get("test/link/smape", 0)
            test_link_node_r2 = test_metrics.get("test/link/node_r2", 0)
            test_link_node_mape = test_metrics.get("test/link/node_mape", 0)
            test_link_node_mae = test_metrics.get("test/link/node_mae", 0)
            test_stop_mae = test_metrics.get("test/stop/mae", 0)
            test_stop_rmse = test_metrics.get("test/stop/rmse", 0)
            test_stop_smape = test_metrics.get("test/stop/smape", 0)
            test_stop_node_r2 = test_metrics.get("test/stop/node_r2", 0)
            test_stop_node_mape = test_metrics.get("test/stop/node_mape", 0)
            test_stop_accuracy = test_metrics.get("test/stop_accuracy", 0)
            test_stop_f1 = test_metrics.get("test/stop_f1", 0)
            test_stop_auc = test_metrics.get("test/stop_auc", 0)

            # Seasonal MASE / RMSSE for test
            test_link_smase_str, test_link_srmsse_str = "N/A", "N/A"
            if test_link_mae > 0:
                test_link_smase_str, test_link_srmsse_str = _compute_smase_metrics(
                    df_travel_test, df_travel_test_full, "link",
                    test_link_mae, test_link_rmse, test_metrics, "test/link",
                )

            test_stop_smase_str, test_stop_srmsse_str = "N/A", "N/A"
            if test_stop_mae > 0:
                test_stop_smase_str, test_stop_srmsse_str = _compute_smase_metrics(
                    df_dwell_test[df_dwell_test["travel_time"] > 0],
                    df_dwell_test_full[df_dwell_test_full["travel_time"] > 0],
                    "stop", test_stop_mae, test_stop_rmse, test_metrics, "test/stop",
                )

            # Compute final oversmoothing metrics (GNN only — XGBoost has no node embeddings)
            if args.model == "xgboost":
                final_oversmoothing = {}
            else:
                final_oversmoothing = compute_oversmoothing_metrics(model, train_graph, device)

            logger.info("Test | loss=%.4f (link=%.4f, stop=%.4f [reg=%.4f, cls=%.4f])",
                       test_loss, test_link_loss, test_stop_loss, test_stop_reg_loss, test_stop_cls_loss)
            logger.info("  Link  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s | MAD: %6.3f | MADGap: %6.3f",
                       test_link_mae, test_link_rmse, test_link_smape,
                       test_link_smase_str, test_link_srmsse_str,
                       final_oversmoothing.get("link/mad", 0), final_oversmoothing.get("link/mad_gap", 0))
            logger.info("  Stop  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s | F1: %6.3f | AUC: %6.3f | Acc: %6.3f | MAD: %6.3f | MADGap: %6.3f",
                       test_stop_mae, test_stop_rmse, test_stop_smape,
                       test_stop_smase_str, test_stop_srmsse_str,
                       test_stop_f1, test_stop_auc, test_stop_accuracy,
                       final_oversmoothing.get("stop/mad", 0), final_oversmoothing.get("stop/mad_gap", 0))

            if run is not None:
                # Best validation loss (for sweep optimization)
                run.summary["best_val_loss"] = best_val_loss
                
                # Test metrics from transfer (OOD) regions
                run.summary.update(split_metrics_by_node_type(test_metrics, "test_transfer", best_prefix=True))
                
                # Test oversmoothing metrics
                run.summary.update(split_oversmoothing_metrics(final_oversmoothing, best_prefix=True))

                # Best validation metrics from transfer (OOD) regions
                if best_val_metrics is not None:
                    best_val_split = split_metrics_by_node_type(best_val_metrics, "val_transfer", best_prefix=True)
                    run.summary.update(best_val_split)
                    
                    # Combined sRMSSE for sweep optimization (geometric mean)
                    best_link_srmsse = best_val_split.get("best_val_transfer_link/srmsse")
                    best_stop_srmsse = best_val_split.get("best_val_transfer_stop/srmsse")
                    if best_link_srmsse is not None and best_stop_srmsse is not None:
                        combined = (best_link_srmsse * best_stop_srmsse) ** 0.5
                        run.summary["best_val_transfer_combined/srmsse"] = combined
                        run.summary["val_transfer_combined/srmsse"] = combined
                
                # Best validation oversmoothing metrics
                if best_oversmoothing_metrics is not None:
                    run.summary.update(split_oversmoothing_metrics(best_oversmoothing_metrics, best_prefix=True))

            # Collect test metrics with clean keys for aggregate summary
            split_test_metrics = {}
            split_test_metrics.update(split_metrics_by_node_type(test_metrics, "test_transfer", best_prefix=True))
            split_test_metrics.update(split_oversmoothing_metrics(final_oversmoothing, best_prefix=True))
            # Also collect val-transfer metrics from best checkpoint
            if best_val_metrics is not None:
                split_test_metrics.update(split_metrics_by_node_type(best_val_metrics, "val_transfer", best_prefix=True))
            all_test_metrics.append(split_test_metrics)
        else:
            logger.info("No test data available for this split (route filter may exclude test regions)")

        # Collect train eval metrics for aggregate summary
        split_train_eval_metrics = split_metrics_by_node_type(train_eval_metrics, "train", best_prefix=True)
        if all_test_metrics:
            all_test_metrics[-1].update(split_train_eval_metrics)
        else:
            all_test_metrics.append(split_train_eval_metrics)

        # Same-region temporal test evaluation (train regions × test time window)
        # Only logged to wandb, measures temporal generalization on known regions
        if train_testtime_loader is not None and train_testtime_dataset is not None:
            logger.info("Evaluating on train regions with test time window (temporal generalization)")
            (
                train_testtime_loss, train_testtime_link_loss,
                train_testtime_stop_reg_loss, train_testtime_stop_cls_loss,
                train_testtime_stop_loss, train_testtime_metrics
            ) = evaluate(
                model, train_graph, train_testtime_loader, device,
                link_reg_weight=link_reg_weight,
                stop_reg_weight=stop_reg_weight,
                stop_cls_weight=stop_cls_weight,
                use_amp=use_amp,
                link_scaler=train_dataset.link_scaler,
                stop_scaler=train_dataset.stop_scaler,
                prefix="train_region_test",
                log_transform=log_transform,
                link_freeflow_refs=train_testtime_dataset.link_freeflow_refs,
                freeflow_ratio=freeflow_ratio,
                link_municipalities=train_testtime_dataset.link_municipalities,
                stop_municipalities=train_testtime_dataset.stop_municipalities,
            )

            tr_test_link_mae = train_testtime_metrics.get("train_region_test/link/mae", 0)
            tr_test_link_rmse = train_testtime_metrics.get("train_region_test/link/rmse", 0)
            tr_test_link_smape = train_testtime_metrics.get("train_region_test/link/smape", 0)
            tr_test_stop_mae = train_testtime_metrics.get("train_region_test/stop/mae", 0)
            tr_test_stop_rmse = train_testtime_metrics.get("train_region_test/stop/rmse", 0)
            tr_test_stop_smape = train_testtime_metrics.get("train_region_test/stop/smape", 0)

            # Seasonal MASE / RMSSE for train-region test
            tr_test_link_smase_str, tr_test_link_srmsse_str = "N/A", "N/A"
            if tr_test_link_mae > 0:
                tr_test_link_smase_str, tr_test_link_srmsse_str = _compute_smase_metrics(
                    df_travel_train_testtime, df_travel_train_full, "link",
                    tr_test_link_mae, tr_test_link_rmse, train_testtime_metrics, "train_region_test/link",
                )

            tr_test_stop_smase_str, tr_test_stop_srmsse_str = "N/A", "N/A"
            if tr_test_stop_mae > 0:
                tr_test_stop_smase_str, tr_test_stop_srmsse_str = _compute_smase_metrics(
                    df_dwell_train_testtime[df_dwell_train_testtime["travel_time"] > 0],
                    df_dwell_train_full[df_dwell_train_full["travel_time"] > 0],
                    "stop", tr_test_stop_mae, tr_test_stop_rmse, train_testtime_metrics, "train_region_test/stop",
                )

            logger.info(
                "Train-region test | loss=%.4f (link=%.4f, stop=%.4f)",
                train_testtime_loss, train_testtime_link_loss, train_testtime_stop_loss,
            )
            logger.info(
                "  Link  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s",
                tr_test_link_mae, tr_test_link_rmse, tr_test_link_smape, tr_test_link_smase_str, tr_test_link_srmsse_str,
            )
            logger.info(
                "  Stop  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s",
                tr_test_stop_mae, tr_test_stop_rmse, tr_test_stop_smape, tr_test_stop_smase_str, tr_test_stop_srmsse_str,
            )

            if run is not None:
                # Test metrics from original (train) regions
                run.summary.update(split_metrics_by_node_type(train_testtime_metrics, "test_original", best_prefix=True))

            # Collect test-original metrics for aggregate summary
            if all_test_metrics:
                all_test_metrics[-1].update(
                    split_metrics_by_node_type(train_testtime_metrics, "test_original", best_prefix=True)
                )

        # Same-region temporal val evaluation at best model (train regions × val time window)
        if train_valtime_loader is not None and train_valtime_dataset is not None:
            logger.info("Evaluating on train regions with val time window (temporal generalization)")
            (
                train_valtime_loss, train_valtime_link_loss,
                train_valtime_stop_reg_loss, train_valtime_stop_cls_loss,
                train_valtime_stop_loss, train_valtime_metrics
            ) = evaluate(
                model, train_graph, train_valtime_loader, device,
                link_reg_weight=link_reg_weight,
                stop_reg_weight=stop_reg_weight,
                stop_cls_weight=stop_cls_weight,
                use_amp=use_amp,
                link_scaler=train_dataset.link_scaler,
                stop_scaler=train_dataset.stop_scaler,
                prefix="train_region_val_final",
                log_transform=log_transform,
                link_freeflow_refs=train_valtime_dataset.link_freeflow_refs,
                freeflow_ratio=freeflow_ratio,
                link_municipalities=train_valtime_dataset.link_municipalities,
                stop_municipalities=train_valtime_dataset.stop_municipalities,
            )

            tr_val_link_mae = train_valtime_metrics.get("train_region_val_final/link/mae", 0)
            tr_val_link_rmse = train_valtime_metrics.get("train_region_val_final/link/rmse", 0)
            tr_val_link_smape = train_valtime_metrics.get("train_region_val_final/link/smape", 0)
            tr_val_stop_mae = train_valtime_metrics.get("train_region_val_final/stop/mae", 0)
            tr_val_stop_rmse = train_valtime_metrics.get("train_region_val_final/stop/rmse", 0)
            tr_val_stop_smape = train_valtime_metrics.get("train_region_val_final/stop/smape", 0)

            # Seasonal MASE / RMSSE for train-region val
            tr_val_link_smase_str, tr_val_link_srmsse_str = "N/A", "N/A"
            if tr_val_link_mae > 0:
                tr_val_link_smase_str, tr_val_link_srmsse_str = _compute_smase_metrics(
                    df_travel_train_valtime, df_travel_train_full, "link",
                    tr_val_link_mae, tr_val_link_rmse, train_valtime_metrics, "train_region_val_final/link",
                )

            tr_val_stop_smase_str, tr_val_stop_srmsse_str = "N/A", "N/A"
            if tr_val_stop_mae > 0:
                tr_val_stop_smase_str, tr_val_stop_srmsse_str = _compute_smase_metrics(
                    df_dwell_train_valtime[df_dwell_train_valtime["travel_time"] > 0],
                    df_dwell_train_full[df_dwell_train_full["travel_time"] > 0],
                    "stop", tr_val_stop_mae, tr_val_stop_rmse, train_valtime_metrics, "train_region_val_final/stop",
                )

            logger.info(
                "Train-region val  | loss=%.4f (link=%.4f, stop=%.4f)",
                train_valtime_loss, train_valtime_link_loss, train_valtime_stop_loss,
            )
            logger.info(
                "  Link  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s",
                tr_val_link_mae, tr_val_link_rmse, tr_val_link_smape, tr_val_link_smase_str, tr_val_link_srmsse_str,
            )
            logger.info(
                "  Stop  - MAE: %7.3f | RMSE: %7.3f | sMAPE: %7.3f%% | sMASE: %s | sRMSSE: %s",
                tr_val_stop_mae, tr_val_stop_rmse, tr_val_stop_smape, tr_val_stop_smase_str, tr_val_stop_srmsse_str,
            )

            if run is not None:
                # Val metrics from original (train) regions
                run.summary.update(split_metrics_by_node_type(train_valtime_metrics, "val_original", best_prefix=True))

            # Collect val-original metrics for aggregate summary
            if all_test_metrics:
                all_test_metrics[-1].update(
                    split_metrics_by_node_type(train_valtime_metrics, "val_original", best_prefix=True)
                )

        # Save model checkpoint
        split_output_dir = os.path.join(args.output_dir, args.model, f"split_{split_idx}")
        os.makedirs(split_output_dir, exist_ok=True)

        model_save_path = os.path.join(split_output_dir, "embedding_model.pt")
        checkpoint: Dict[str, Any] = {
            "model_type": args.model,
            "hyperparams": hyperparams,
            "best_val_loss": best_val_loss,
            "split_idx": split_idx,
            "link_scaler": train_dataset.link_scaler,
            "stop_scaler": train_dataset.stop_scaler,
            "log_transform": log_transform,
            "feature_norm_stats": train_norm_stats,
            "travel_time_residual_mode": travel_time_residual_mode,
            "freeflow_times": freeflow_times_train,
            "train_regions": train_regions,
            "val_regions": val_regions,
            "test_regions": test_regions,
        }
        if args.model == "xgboost":
            # Wrapper is picklable (xgboost models pickle natively).
            checkpoint["model"] = model
        else:
            link_features = train_graph["link"].x.size(-1)
            stop_features = train_graph["stop"].x.size(-1)
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["model_config"] = {
                "link_features_in": link_features,
                "stop_features_in": stop_features,
                "gnn-hidden-dim": hyperparams["gnn-hidden-dim"],
                "gnn-output-dim": hyperparams["gnn-output-dim"],
                "gnn-layers": hyperparams["gnn-layers"],
                "dropout": hyperparams["dropout"],
                "head-hidden-dim": hyperparams["head-hidden-dim"],
                "head-type": hyperparams.get("head-type", "film"),
                "head-time-concat": hyperparams.get("head-time-concat", True),
                **({
                    "num-heads": hyperparams["num-heads"],
                    "attn-dropout": hyperparams.get("attn-dropout"),
                } if args.model == "gat" else {}),
            }
        torch.save(checkpoint, model_save_path)
        logger.info("Model saved to %s", model_save_path)

        if run is not None:
            run.finish()

        # Cleanup
        del model, train_loader
        if val_loader is not None:
            del val_loader
        if test_loader is not None:
            del test_loader
        if train_valtime_loader is not None:
            del train_valtime_loader
        if train_testtime_loader is not None:
            del train_testtime_loader
        del train_dataset
        if val_dataset is not None:
            del val_dataset
        if test_dataset is not None:
            del test_dataset
        if train_valtime_dataset is not None:
            del train_valtime_dataset
        if train_testtime_dataset is not None:
            del train_testtime_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate metrics across splits using geometric mean
    logger.info("=" * 60)
    logger.info("AGGREGATE METRICS (geometric mean across %d splits)", len(splits_to_train))
    logger.info("=" * 60)

    aggregated_metrics: Dict[str, float] = {}
    if all_test_metrics:
        # Collect all metric keys
        metric_keys = set()
        for m in all_test_metrics:
            metric_keys.update(m.keys())

        for key in sorted(metric_keys):
            values = [m.get(key, 0) for m in all_test_metrics if key in m]
            if values:
                # Use arithmetic mean for metrics that can be negative (MADGap, bias, R²)
                use_arith = (
                    "mad_gap" in key or "bias" in key or "node_r2" in key
                    or any(v < 0 for v in values)
                )
                if use_arith:
                    agg_value = sum(values) / len(values)
                    agg_type = "mean"
                else:
                    agg_value = compute_geometric_mean(values)
                    agg_type = "geomean"
                aggregated_metrics[f"agg/{key}"] = agg_value
                # Extract short name for logging
                short_key = key.split("/")[-1]
                logger.info("  %s: %.4f (%s of %s)", short_key, agg_value, agg_type, values)

    # Log aggregated metrics to wandb (skip in sweep mode)
    if not args.no_wandb and all_test_metrics and not args.sweep:
        # Create a summary run for aggregated metrics
        summary_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=group_name,
            name=f"agg_{group_name}",
            config={
                "model_type": args.model,
                "stage": 1,
                "aggregation": "geometric_mean",
                "n_splits": len(splits_to_train),
                "train_regions": train_regions,
                "val_regions": val_regions,
                "test_regions": test_regions,
            },
            reinit=True,
        )

        # Define metrics for aggregated run (no step metric since it's a single summary)
        # All aggregated metrics use geometric mean across splits
        wandb.define_metric("agg/*", summary="last")
        wandb.define_metric("best_val_loss", summary="last")

        # Log aggregated metrics
        summary_run.log(aggregated_metrics)
        summary_run.summary["best_val_loss"] = compute_geometric_mean(all_val_losses)
        summary_run.summary.update(aggregated_metrics)
        summary_run.finish()

    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("Output: %s/%s/split_*/", args.output_dir, args.model)


if __name__ == "__main__":
    main()
