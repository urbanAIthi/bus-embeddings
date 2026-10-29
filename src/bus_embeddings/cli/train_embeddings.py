"""Train GNN embeddings of a public transport network on the proxy tasks.

Trains a GraphSAGE or GATv2 encoder with prediction heads for link travel
times, stop dwell times and stop probabilities, or (``--model xgboost``) the
XGBoost baseline on the same features. Every model is trained on five
expanding temporal folds; within each fold, the municipalities are split
into training (A), validation (B) and test (C) regions. The checkpoint with
the best validation score of each fold is evaluated on five subsets:

    train          regions A, training period
    val_temporal   regions A, validation period
    val_spatial    regions B, validation period
    test_temporal  regions A, test period
    test_spatial   regions C, test period

Writes one checkpoint per fold (``fold_<i>.pt``) and ``metrics.json`` with the
per-fold metrics and their aggregate over folds to ``--output-dir``.

Example::

    python -m bus_embeddings.cli.train_embeddings \\
        --artifacts-dir data/artifacts \\
        --travel-times-parquet data/preprocessed/travel_times.parquet \\
        --dwell-times-parquet data/preprocessed/dwell_times.parquet \\
        --model gat --output-dir results/gat
"""
import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData

from bus_embeddings.data.proxy_task_dataset import (
    GlobalScaler,
    ProxyTaskBatchIterator,
    RawObservationProxyDataset,
    load_observations,
)
from bus_embeddings.logging import get_logger, setup_logging
from bus_embeddings.models.embedding_gnn import BaseEmbeddingGNN, build_model
from bus_embeddings.models.xgboost_baseline import XGBoostBaseline
from bus_embeddings.training.config import MODEL_DEFAULTS, SEED, STOP_CLASSIFIER_THRESHOLDS
from bus_embeddings.training.metrics import (
    compute_oversmoothing_metrics,
    compute_seasonal_naive_metrics,
    compute_seasonal_naive_metrics_by_region,
)
from bus_embeddings.training.train_utils import compute_geometric_mean, seed_all, uses_ratio_target
from bus_embeddings.utils.hetero_graph_utils import build_hetero_graph
from bus_embeddings.utils.splits import (
    assign_regions_to_splits,
    discover_regions,
    fold_boundaries,
    region_artifact_paths,
)

logger = get_logger(__name__)

# Evaluation subsets: (region group, period)
SUBSETS: Dict[str, Tuple[str, str]] = {
    "train": ("train", "train"),
    "val_temporal": ("train", "val"),
    "val_spatial": ("val", "val"),
    "test_temporal": ("train", "test"),
    "test_spatial": ("test", "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GNN embeddings on the travel time, dwell time and stop proxy tasks.",
    )
    parser.add_argument(
        "--artifacts-dir", required=True,
        help="Directory with one sub-directory of graph artifacts (nodes_link.csv, "
             "nodes_stop.csv, edges.csv) per municipality.",
    )
    parser.add_argument(
        "--travel-times-parquet", required=True,
        help="Preprocessed travel times (output of preprocess_data).",
    )
    parser.add_argument(
        "--dwell-times-parquet", required=True,
        help="Preprocessed dwell times (output of preprocess_data).",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for the checkpoints and metrics.json (default: results).",
    )
    parser.add_argument(
        "--model", choices=sorted(MODEL_DEFAULTS), default="gat",
        help="Model to train (default: gat).",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a hyperparameter of the model (see training/config.py); repeatable. "
             "Values are parsed as JSON, e.g. --set global-skip-connection=false --set gnn-layers=0.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED}).")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument("--wandb-project", default="bus-embeddings", help="Weights & Biases project.")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity.")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )
    return parser.parse_args()


def apply_overrides(hyperparams: Dict[str, Any], overrides: List[str]) -> None:
    """Apply ``KEY=VALUE`` overrides (values parsed as JSON) to ``hyperparams``."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid override {item!r}, expected KEY=VALUE")
        if key not in hyperparams:
            raise ValueError(f"Unknown hyperparameter {key!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        logger.info("Override: %s = %r (default %r)", key, value, hyperparams[key])
        hyperparams[key] = value


# =============================================================================
# Training and evaluation
# =============================================================================


def train_epoch(
    model: BaseEmbeddingGNN,
    graph: HeteroData,
    loader: ProxyTaskBatchIterator,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    hyperparams: Dict[str, Any],
    scaler: Optional[GradScaler],
) -> Dict[str, float]:
    """Train for one epoch and return the mean training losses."""
    link_w = hyperparams["link-reg-weight"]
    stop_reg_w = hyperparams["stop-reg-weight"]
    stop_cls_w = hyperparams["stop-cls-weight"]
    huber_delta = hyperparams["huber-delta"]
    focal_gamma = hyperparams["focal-gamma"]

    model.train()
    # Accumulate on the GPU to avoid synchronisation
    total_link_loss = torch.tensor(0.0, device=device)
    total_stop_reg_loss = torch.tensor(0.0, device=device)
    total_stop_cls_loss = torch.tensor(0.0, device=device)
    n_link = n_stop = n_stopped = 0

    use_amp = scaler is not None and device.type == "cuda"
    graph = graph.to(device)

    for batch in loader:
        link_idx, link_ctx, link_y, stop_idx, stop_ctx, stop_y, stop_labels = (
            t.to(device) if t is not None else None for t in batch
        )
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=use_amp):
            link_preds, stop_preds, stop_logits, _ = model(
                graph, link_idx, link_ctx, stop_idx, stop_ctx,
            )

            link_loss = torch.tensor(0.0, device=device)
            if link_preds.numel() > 0:
                link_loss = F.huber_loss(link_preds, link_y, reduction="mean", delta=huber_delta)

            stop_reg_loss = torch.tensor(0.0, device=device)
            stop_cls_loss = torch.tensor(0.0, device=device)
            if stop_preds.numel() > 0 and stop_labels is not None:
                # Focal loss for the stop classification
                bce = F.binary_cross_entropy_with_logits(stop_logits, stop_labels, reduction="none")
                p_t = torch.exp(-bce)
                stop_cls_loss = ((1 - p_t) ** focal_gamma * bce).mean()

                # Dwell time regression only where the bus stopped
                stopped = stop_labels > 0.5
                if stopped.any():
                    stop_reg_loss = F.huber_loss(
                        stop_preds[stopped], stop_y[stopped], reduction="mean", delta=huber_delta,
                    )

            loss = (
                link_w * link_loss + stop_cls_w * stop_cls_loss + stop_reg_w * stop_reg_loss
            ) / (link_w + stop_cls_w + stop_reg_w)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams["grad-clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams["grad-clip"])
            optimizer.step()

        total_link_loss += link_loss.detach() * link_preds.numel()
        n_link += link_preds.numel()
        if stop_preds.numel() > 0 and stop_labels is not None:
            total_stop_cls_loss += stop_cls_loss.detach() * stop_preds.numel()
            n_stop += stop_preds.numel()
            stopped = stop_labels > 0.5
            if stopped.any():
                total_stop_reg_loss += stop_reg_loss.detach() * stopped.sum()
                n_stopped += stopped.sum().item()

    link_loss_avg = (total_link_loss / max(n_link, 1)).item()
    stop_reg_avg = (total_stop_reg_loss / max(n_stopped, 1)).item()
    stop_cls_avg = (total_stop_cls_loss / max(n_stop, 1)).item()
    return {
        "loss": (link_w * link_loss_avg + stop_cls_w * stop_cls_avg + stop_reg_w * stop_reg_avg)
        / (link_w + stop_cls_w + stop_reg_w),
        "link_loss": link_loss_avg,
        "stop_reg_loss": stop_reg_avg,
        "stop_cls_loss": stop_cls_avg,
    }


@dataclass
class EvalResult:
    """Losses, metrics and per-observation predictions (seconds) of one evaluation."""

    losses: Dict[str, float]
    metrics: Dict[str, float]
    link_preds: np.ndarray = field(repr=False)
    link_targets: np.ndarray = field(repr=False)
    stop_preds: np.ndarray = field(repr=False)
    stop_targets: np.ndarray = field(repr=False)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    graph: HeteroData,
    dataset: RawObservationProxyDataset,
    device: torch.device,
    hyperparams: Dict[str, Any],
    link_scaler: GlobalScaler,
    stop_scaler: GlobalScaler,
    ratio_target: bool,
) -> EvalResult:
    """Evaluate on all observations of ``dataset``.

    Computes MAE and RMSE in seconds for travel times and for the dwell times
    of stopped buses, and F1 for the stop classification. The validation
    losses (MSE / BCE on the normalized targets) are only used for logging.
    """
    model.eval()
    loader = ProxyTaskBatchIterator(
        dataset, batch_size=hyperparams["batch-size"], pin_memory=device.type == "cuda",
    )
    graph = graph.to(device)
    amp_enabled = hyperparams.get("use-amp", False) and device.type == "cuda"
    log_transform = hyperparams["log-transform"]

    total_link_loss = total_stop_reg_loss = total_stop_cls_loss = 0.0
    n_link = n_stop = n_stopped = 0
    link_preds_all, link_targets_all = [], []
    stop_preds_all, stop_targets_all = [], []
    stop_logits_all, stop_labels_all = [], []

    for batch in loader:
        link_idx, link_ctx, link_y, stop_idx, stop_ctx, stop_y, stop_labels = (
            t.to(device) if t is not None else None for t in batch
        )
        with autocast(device_type="cuda", enabled=amp_enabled):
            link_preds, stop_preds, stop_logits, _ = model(
                graph, link_idx, link_ctx, stop_idx, stop_ctx,
            )
        if amp_enabled:
            link_preds = link_preds.float()
            stop_preds = stop_preds.float()
            stop_logits = stop_logits.float()

        if link_preds.numel() > 0:
            total_link_loss += F.mse_loss(link_preds, link_y, reduction="sum").item()
            n_link += link_preds.numel()
            link_preds_all.append(link_preds.cpu())
            link_targets_all.append(link_y.cpu())

        if stop_preds.numel() > 0 and stop_labels is not None:
            total_stop_cls_loss += F.binary_cross_entropy_with_logits(
                stop_logits, stop_labels, reduction="sum"
            ).item()
            n_stop += stop_preds.numel()
            stopped = stop_labels > 0.5
            if stopped.any():
                total_stop_reg_loss += F.mse_loss(
                    stop_preds[stopped], stop_y[stopped], reduction="sum"
                ).item()
                n_stopped += stopped.sum().item()
            # Dwell time metrics only on stopped observations
            stop_preds_all.append(stop_preds[stopped].cpu())
            stop_targets_all.append(stop_y[stopped].cpu())
            stop_logits_all.append(stop_logits.cpu())
            stop_labels_all.append(stop_labels.cpu())

    link_w = hyperparams.get("link-reg-weight", 1.0)
    stop_reg_w = hyperparams.get("stop-reg-weight", 1.0)
    stop_cls_w = hyperparams.get("stop-cls-weight", 1.0)
    link_loss = total_link_loss / max(n_link, 1)
    stop_reg_loss = total_stop_reg_loss / max(n_stopped, 1)
    stop_cls_loss = total_stop_cls_loss / max(n_stop, 1)
    losses = {
        "loss": (link_w * link_loss + stop_cls_w * stop_cls_loss + stop_reg_w * stop_reg_loss)
        / (link_w + stop_cls_w + stop_reg_w),
        "link_loss": link_loss,
        "stop_reg_loss": stop_reg_loss,
        "stop_cls_loss": stop_cls_loss,
    }

    metrics: Dict[str, float] = {}
    link_preds_sec = link_targets_sec = np.empty(0, dtype=np.float32)
    stop_preds_sec = stop_targets_sec = np.empty(0, dtype=np.float32)

    if link_preds_all:
        preds = link_scaler.inverse_transform_tensor(torch.cat(link_preds_all))
        targets = link_scaler.inverse_transform_tensor(torch.cat(link_targets_all))
        # Undo the target transformation relative to the ideal travel time
        refs = dataset.link_freeflow_refs
        if len(preds) != len(refs):
            raise RuntimeError(f"{len(preds)} predictions but {len(refs)} ideal travel times")
        if ratio_target:
            preds, targets = preds * refs, targets * refs
        else:
            preds, targets = preds + refs, targets + refs
        if log_transform:
            preds, targets = torch.expm1(preds), torch.expm1(targets)
        preds = preds.clamp(min=0.0)  # travel times cannot be negative

        metrics["link/mae"] = (preds - targets).abs().mean().item()
        metrics["link/rmse"] = ((preds - targets) ** 2).mean().sqrt().item()
        link_preds_sec, link_targets_sec = preds.numpy(), targets.numpy()

    if stop_preds_all:
        preds = stop_scaler.inverse_transform_tensor(torch.cat(stop_preds_all))
        targets = stop_scaler.inverse_transform_tensor(torch.cat(stop_targets_all))
        if log_transform:
            preds, targets = torch.expm1(preds), torch.expm1(targets)

        metrics["stop/mae"] = (preds - targets).abs().mean().item()
        metrics["stop/rmse"] = ((preds - targets) ** 2).mean().sqrt().item()
        stop_preds_sec, stop_targets_sec = preds.numpy(), targets.numpy()

    if stop_logits_all:
        probs = torch.sigmoid(torch.cat(stop_logits_all))
        labels = torch.cat(stop_labels_all)  # 1 = stopped, 0 = passed through
        threshold = STOP_CLASSIFIER_THRESHOLDS.get(getattr(model, "MODEL_TYPE", ""), 0.5)
        predicted = (probs > threshold).float()

        true_positives = ((predicted == 1) & (labels == 1)).sum().float()
        predicted_positives = (predicted == 1).sum().float()
        actual_positives = (labels == 1).sum().float()
        precision = (true_positives / predicted_positives).item() if predicted_positives > 0 else 0.0
        recall = (true_positives / actual_positives).item() if actual_positives > 0 else 0.0
        metrics["stop/precision"] = precision
        metrics["stop/recall"] = recall
        metrics["stop/f1"] = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )

    return EvalResult(losses, metrics, link_preds_sec, link_targets_sec, stop_preds_sec, stop_targets_sec)


def add_scaled_errors(
    result: EvalResult,
    dataset: RawObservationProxyDataset,
    observations: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    per_region: bool = True,
) -> None:
    """Add seasonal MASE and RMSSE of links and stops to ``result.metrics``.

    ``observations`` maps ``"link"`` and ``"stop"`` to the evaluated
    observations and the history the seasonal naive forecast is taken from.
    With ``per_region`` the ratios are computed per municipality and combined
    with the geometric mean; otherwise they are pooled over all municipalities.
    """
    for node_type, preds, targets, regions in (
        ("link", result.link_preds, result.link_targets, dataset.link_municipalities),
        ("stop", result.stop_preds, result.stop_targets, dataset.stop_municipalities),
    ):
        if result.metrics.get(f"{node_type}/mae", 0) <= 0:
            continue
        df_eval, df_history = observations[node_type]
        if per_region:
            if regions is None:
                raise ValueError("Observations have no municipality column")
            scaled = compute_seasonal_naive_metrics_by_region(
                df_eval, df_history, node_type, preds, targets, regions,
            )
        else:
            scaled = compute_seasonal_naive_metrics(
                df_eval, df_history, node_type,
                model_mae=result.metrics[f"{node_type}/mae"],
                model_rmse=result.metrics[f"{node_type}/rmse"],
            )
        for metric in ("mase", "rmsse"):
            if metric in scaled:
                result.metrics[f"{node_type}/{metric}"] = scaled[metric]


def validation_score(result: EvalResult) -> float:
    """Model selection score: geometric mean of the link and stop RMSSE."""
    link_rmsse = result.metrics.get("link/rmsse")
    stop_rmsse = result.metrics.get("stop/rmsse")
    if link_rmsse is None or stop_rmsse is None:
        return result.losses["loss"]
    return (link_rmsse * stop_rmsse) ** 0.5


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["link/mase", "link/rmsse", "stop/mase", "stop/rmsse", "stop/f1"]
    return " | ".join(f"{k} {metrics[k]:.4f}" for k in keys if k in metrics)


def train_gnn(
    model_type: str,
    hyperparams: Dict[str, Any],
    device: torch.device,
    train_graph: HeteroData,
    val_graph: HeteroData,
    train_dataset: RawObservationProxyDataset,
    val_dataset: RawObservationProxyDataset,
    val_observations: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ratio_target: bool,
    run: Any,
) -> Tuple[BaseEmbeddingGNN, EvalResult]:
    """Train a GNN with early stopping on the validation regions.

    Returns the model with the best validation score and its validation result.
    """
    model = build_model(
        model_type, hyperparams,
        link_features_in=train_graph["link"].x.size(-1),
        stop_features_in=train_graph["stop"].x.size(-1),
    ).to(device)
    logger.info("Model: %s (%s parameters)", model_type,
                f"{sum(p.numel() for p in model.parameters()):,}")

    # Separate learning rate and weight decay for the GNN and the prediction heads
    head_modules = ("link_head", "stop_head", "stop_classifier", "embedding_norm")
    gnn_params, head_params = [], []
    for name, param in model.named_parameters():
        (head_params if name.startswith(head_modules) else gnn_params).append(param)
    base_lr = hyperparams["learning-rate"]
    optimizer = torch.optim.AdamW([
        {"params": gnn_params, "lr": base_lr * hyperparams["gnn-lr-factor"],
         "weight_decay": hyperparams["gnn-weight-decay"]},
        {"params": head_params, "lr": base_lr, "weight_decay": hyperparams["weight-decay"]},
    ])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hyperparams["scheduler-factor"],
        patience=hyperparams["scheduler-patience"],
        min_lr=hyperparams["min-lr"],
    )
    use_amp = hyperparams["use-amp"] and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    # Every epoch trains on a fresh random subsample of the observations
    train_loader = ProxyTaskBatchIterator(
        train_dataset,
        batch_size=hyperparams["batch-size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
        subsample=hyperparams["train-subsample"],
    )

    best_score = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_result: Optional[EvalResult] = None
    epochs_without_improvement = 0
    patience = hyperparams["early-stopping-patience"]

    for epoch in range(1, hyperparams["epochs"] + 1):
        train_losses = train_epoch(
            model, train_graph, train_loader, optimizer, device, hyperparams, scaler,
        )
        val = evaluate(
            model, val_graph, val_dataset, device, hyperparams,
            train_dataset.link_scaler, train_dataset.stop_scaler, ratio_target,
        )
        add_scaled_errors(val, val_dataset, val_observations)
        score = validation_score(val)

        logger.info(
            "Epoch %03d | train loss %.4f | val loss %.4f | %s",
            epoch, train_losses["loss"], val.losses["loss"], format_metrics(val.metrics),
        )
        if run is not None:
            run.log({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train/{k}": v for k, v in train_losses.items()},
                **{f"val_spatial/{k}": v for k, v in {**val.losses, **val.metrics}.items()},
                "val_spatial/score": score,
            })

        scheduler.step(score)
        if score < best_score - hyperparams["early-stopping-min-delta"]:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_result = val
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    assert best_state is not None and best_result is not None
    model.load_state_dict(best_state)
    return model, best_result


# =============================================================================
# Main
# =============================================================================


@dataclass
class FoldData:
    """Observations and graph nodes per region group, and the periods of one fold."""

    travel: Dict[str, pd.DataFrame]
    dwell: Dict[str, pd.DataFrame]
    link_nodes: Dict[str, pd.DataFrame]
    stop_nodes: Dict[str, pd.DataFrame]
    freeflow_times: Dict[str, Dict[str, float]]
    periods: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]
    ratio_target: bool
    log_transform: bool

    def window(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        start, end = self.periods[period]
        return df[(df["from_time"] >= start) & (df["from_time"] < end)]

    def observations(self, subset: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Evaluated observations of a subset and the history for the seasonal naive forecast."""
        group, period = SUBSETS[subset]
        stops = self.dwell[group][self.dwell[group]["travel_time"] > 0]
        return {
            "link": (self.window(self.travel[group], period), self.travel[group]),
            "stop": (self.window(stops, period), stops),
        }

    def dataset(
        self, subset: str, fitted: Optional[RawObservationProxyDataset] = None,
    ) -> RawObservationProxyDataset:
        """Samples of a subset; target scalers are taken from ``fitted`` or fitted here."""
        group, period = SUBSETS[subset]
        return RawObservationProxyDataset(
            df_travel=self.window(self.travel[group], period),
            df_dwell=self.window(self.dwell[group], period),
            link_nodes=self.link_nodes[group],
            stop_nodes=self.stop_nodes[group],
            freeflow_times=self.freeflow_times[group],
            freeflow_ratio=self.ratio_target,
            log_transform=self.log_transform,
            link_scaler=fitted.link_scaler if fitted is not None else None,
            stop_scaler=fitted.stop_scaler if fitted is not None else None,
        )


def aggregate_folds(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Geometric mean over folds (arithmetic mean for MADGap and negative values)."""
    aggregated: Dict[str, float] = {}
    for key in sorted({k for m in fold_metrics for k in m}):
        values = [m[key] for m in fold_metrics if key in m]
        if "mad_gap" in key or any(v < 0 for v in values):
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = compute_geometric_mean(values)
    return aggregated


def init_wandb(args: argparse.Namespace, name: str, group: str, config: Dict[str, Any]) -> Any:
    try:
        import wandb
    except ImportError as e:
        raise ImportError("--wandb requires the wandb package (install the 'wandb' extra)") from e
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=name,
        config=config,
        reinit=True,
    )


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    hyperparams = dict(MODEL_DEFAULTS[args.model])
    apply_overrides(hyperparams, args.overrides)
    ratio_target = uses_ratio_target(hyperparams)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data and graphs
    # ------------------------------------------------------------------
    df_travel_all, df_dwell_all = load_observations(
        args.travel_times_parquet, args.dwell_times_parquet,
    )
    municipalities = set(df_travel_all["municipality"].unique()) & set(df_dwell_all["municipality"].unique())
    train_regions, val_regions, test_regions = assign_regions_to_splits(
        discover_regions(args.artifacts_dir, municipalities)
    )
    regions = {"train": train_regions, "val": val_regions, "test": test_regions}

    # Observations per region group
    travel = {g: df_travel_all[df_travel_all["municipality"].isin(r)] for g, r in regions.items()}
    dwell = {g: df_dwell_all[df_dwell_all["municipality"].isin(r)] for g, r in regions.items()}
    del df_travel_all, df_dwell_all

    # Validation and test graphs are normalized with the training statistics
    graphs: Dict[str, HeteroData] = {}
    link_nodes: Dict[str, pd.DataFrame] = {}
    stop_nodes: Dict[str, pd.DataFrame] = {}
    norm_stats = None
    for group in ("train", "val", "test"):
        graphs[group], link_nodes[group], stop_nodes[group], stats = build_hetero_graph(
            *region_artifact_paths(regions[group], args.artifacts_dir),
            feature_config=hyperparams,
            feature_norm_stats=norm_stats,
        )
        if norm_stats is None:
            norm_stats = stats
    freeflow_times = {
        group: dict(zip(nodes["id"].astype(str), nodes["osm_travel_time"].astype(float)))
        for group, nodes in link_nodes.items()
    }

    boundaries = fold_boundaries(
        pd.concat([travel["train"], travel["val"], travel["test"]]),
        pd.concat([dwell["train"], dwell["val"], dwell["test"]]),
    )

    group_name = f"{args.model}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}"
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_start, train_end, val_end, test_end) in enumerate(boundaries):
        # Reseed per fold so that each fold is reproducible on its own
        seed_all(args.seed + fold)
        data = FoldData(
            travel, dwell, link_nodes, stop_nodes, freeflow_times,
            periods={"train": (train_start, train_end), "val": (train_end, val_end), "test": (val_end, test_end)},
            ratio_target=ratio_target,
            log_transform=hyperparams["log-transform"],
        )
        logger.info("=" * 60)
        logger.info("FOLD %d/%d: train %s to %s, val to %s, test to %s",
                    fold + 1, len(boundaries), train_start, train_end, val_end, test_end)

        run = None
        if args.wandb:
            run = init_wandb(args, f"fold{fold}_{group_name}", group_name, {
                **hyperparams, "model": args.model, "fold": fold, "seed": args.seed, **regions,
            })

        # Target scalers are fitted on the training subset
        train_dataset = data.dataset("train")
        val_dataset = data.dataset("val_spatial", fitted=train_dataset)

        model: nn.Module
        if args.model == "xgboost":
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
            model.fit(train_dataset, graphs["train"], val_dataset, graphs["val"])
            val_result = evaluate(
                model, graphs["val"], val_dataset, device, hyperparams,
                train_dataset.link_scaler, train_dataset.stop_scaler, ratio_target,
            )
            add_scaled_errors(val_result, val_dataset, data.observations("val_spatial"))
        else:
            model, val_result = train_gnn(
                args.model, hyperparams, device, graphs["train"], graphs["val"],
                train_dataset, val_dataset, data.observations("val_spatial"), ratio_target, run,
            )

        # Evaluate the selected model on all subsets
        results = {"val_spatial": val_result}
        for subset in ("train", "val_temporal", "test_temporal", "test_spatial"):
            dataset = train_dataset if subset == "train" else data.dataset(subset, fitted=train_dataset)
            result = evaluate(
                model, graphs[SUBSETS[subset][0]], dataset, device, hyperparams,
                train_dataset.link_scaler, train_dataset.stop_scaler, ratio_target,
            )
            # The scaled errors of the training subset are pooled over
            # municipalities; all other subsets use the per-region geometric mean.
            add_scaled_errors(result, dataset, data.observations(subset), per_region=subset != "train")
            results[subset] = result
            del dataset

        metrics = {
            f"{subset}/{key}": value
            for subset in SUBSETS
            for key, value in results[subset].metrics.items()
        }
        if args.model != "xgboost":
            for key, value in compute_oversmoothing_metrics(model, graphs["train"], device).items():
                metrics[f"embedding/{key}"] = value
        fold_metrics.append(metrics)

        for subset in SUBSETS:
            logger.info("  %-13s | %s", subset, format_metrics(results[subset].metrics))
        if "embedding/link/mad" in metrics:
            logger.info(
                "  embeddings    | link MAD %.4f MADGap %.4f | stop MAD %.4f MADGap %.4f",
                metrics["embedding/link/mad"], metrics["embedding/link/mad_gap"],
                metrics["embedding/stop/mad"], metrics["embedding/stop/mad_gap"],
            )

        checkpoint: Dict[str, Any] = {
            "model_type": args.model,
            "hyperparams": hyperparams,
            "fold": fold,
            **{f"{group}_regions": r for group, r in regions.items()},
            "feature_norm_stats": norm_stats,
            "link_scaler": train_dataset.link_scaler,
            "stop_scaler": train_dataset.stop_scaler,
            "link_features_in": graphs["train"]["link"].x.size(-1),
            "stop_features_in": graphs["train"]["stop"].x.size(-1),
        }
        if args.model == "xgboost":
            checkpoint["model"] = model
        else:
            checkpoint["model_state_dict"] = model.state_dict()
        checkpoint_path = output_dir / f"fold_{fold}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved %s", checkpoint_path)

        if run is not None:
            run.summary.update(metrics)
            run.finish()

        del model, train_dataset, val_dataset, results
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregated = aggregate_folds(fold_metrics)
    logger.info("=" * 60)
    logger.info("AGGREGATE OVER %d FOLDS", len(fold_metrics))
    for subset in SUBSETS:
        logger.info("  %-13s | %s", subset, format_metrics(
            {k.split("/", 1)[1]: v for k, v in aggregated.items() if k.startswith(subset + "/")}
        ))
    for key in ("link/mad", "link/mad_gap", "stop/mad", "stop/mad_gap"):
        if f"embedding/{key}" in aggregated:
            logger.info("  embedding %s: %.4f", key, aggregated[f"embedding/{key}"])

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model": args.model,
            "seed": args.seed,
            "hyperparams": hyperparams,
            "regions": regions,
            "folds": fold_metrics,
            "aggregate": aggregated,
        }, f, indent=2)
    logger.info("Saved %s", metrics_path)

    if args.wandb:
        run = init_wandb(args, f"agg_{group_name}", group_name, {
            **hyperparams, "model": args.model, "seed": args.seed, **regions,
        })
        run.summary.update(aggregated)
        run.finish()


if __name__ == "__main__":
    main()
