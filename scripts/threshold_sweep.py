#!/usr/bin/env python3
"""Derive the decision threshold of the stop classifier from trained models.

For the checkpoint of each fold, finds the threshold that maximizes the F1
score of the stop classification on the validation regions (val_spatial),
and reports the F1 score on the validation and test regions at 0.5, at the
configured threshold (``STOP_CLASSIFIER_THRESHOLDS``) and at the fold's
optimal threshold. The configured thresholds are the mean optimal threshold
over the five folds.

Usage::

    python scripts/threshold_sweep.py \\
        --checkpoint-dir results/gat \\
        --artifacts-dir data/artifacts \\
        --travel-times-parquet data/preprocessed/travel_times.parquet \\
        --dwell-times-parquet data/preprocessed/dwell_times.parquet
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from bus_embeddings.data.proxy_task_dataset import (
    ProxyTaskBatchIterator,
    RawObservationProxyDataset,
    load_observations,
)
from bus_embeddings.logging import get_logger, setup_logging
from bus_embeddings.models.checkpoint import load_checkpoint
from bus_embeddings.training.config import N_FOLDS, STOP_CLASSIFIER_THRESHOLDS
from bus_embeddings.training.train_utils import uses_ratio_target
from bus_embeddings.utils.hetero_graph_utils import build_hetero_graph
from bus_embeddings.utils.splits import fold_boundaries, region_artifact_paths

logger = get_logger(__name__)


@torch.no_grad()
def stop_probabilities(
    model: torch.nn.Module,
    graph,
    dataset: RawObservationProxyDataset,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stop probability and stop label of every stop observation."""
    loader = ProxyTaskBatchIterator(dataset, batch_size=batch_size, pin_memory=device.type == "cuda")
    graph = graph.to(device)
    logits, labels = [], []
    for link_idx, link_ctx, _, stop_idx, stop_ctx, _, stop_labels in loader:
        if stop_labels is None:
            continue
        _, _, stop_logits, _ = model(
            graph, link_idx.to(device), link_ctx.to(device), stop_idx.to(device), stop_ctx.to(device),
        )
        logits.append(stop_logits.float().cpu().numpy())
        labels.append(stop_labels.numpy())
    logits_all, labels_all = np.concatenate(logits), np.concatenate(labels)
    return 1.0 / (1.0 + np.exp(-logits_all)), labels_all


def f1_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    preds = (probs > threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """F1-optimal threshold: coarse grid search followed by a finer grid around the best value."""
    best_tau, best_f1 = 0.5, 0.0
    for tau in np.linspace(0.05, 0.95, 19):
        f1 = f1_at(probs, labels, tau)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    for tau in np.linspace(max(0.01, best_tau - 0.05), min(0.99, best_tau + 0.05), 21):
        f1 = f1_at(probs, labels, tau)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau


def evaluate_fold(
    checkpoint_path: Path,
    df_travel: pd.DataFrame,
    df_dwell: pd.DataFrame,
    artifacts_dir: str,
    device: torch.device,
) -> Dict[str, float]:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    hyperparams = checkpoint["hyperparams"]
    ratio_target = uses_ratio_target(hyperparams)

    regions = checkpoint["train_regions"] + checkpoint["val_regions"] + checkpoint["test_regions"]
    _, train_end, val_end, test_end = fold_boundaries(
        df_travel[df_travel["municipality"].isin(regions)],
        df_dwell[df_dwell["municipality"].isin(regions)],
    )[checkpoint["fold"]]
    periods = {"val": (train_end, val_end), "test": (val_end, test_end)}

    def subset_dataset(group: str) -> Tuple[object, RawObservationProxyDataset]:
        group_regions = checkpoint[f"{group}_regions"]
        start, end = periods[group]
        graph, link_nodes, stop_nodes, _ = build_hetero_graph(
            *region_artifact_paths(group_regions, artifacts_dir),
            feature_config=hyperparams,
            feature_norm_stats=checkpoint["feature_norm_stats"],
        )

        def window(df: pd.DataFrame) -> pd.DataFrame:
            return df[
                df["municipality"].isin(group_regions)
                & (df["from_time"] >= start)
                & (df["from_time"] < end)
            ]

        dataset = RawObservationProxyDataset(
            df_travel=window(df_travel),
            df_dwell=window(df_dwell),
            link_nodes=link_nodes,
            stop_nodes=stop_nodes,
            freeflow_times=dict(zip(link_nodes["id"].astype(str), link_nodes["osm_travel_time"].astype(float))),
            freeflow_ratio=ratio_target,
            log_transform=hyperparams["log-transform"],
            link_scaler=checkpoint["link_scaler"],
            stop_scaler=checkpoint["stop_scaler"],
        )
        return graph, dataset

    results: Dict[str, float] = {"fold": checkpoint["fold"]}
    configured = STOP_CLASSIFIER_THRESHOLDS.get(checkpoint["model_type"], 0.5)
    probs = {}
    for group in ("val", "test"):
        graph, dataset = subset_dataset(group)
        probs[group] = stop_probabilities(model, graph, dataset, device, hyperparams["batch-size"])
    tau = best_threshold(*probs["val"])
    results["best_threshold"] = tau
    for group in ("val", "test"):
        for label, threshold in (("0.5", 0.5), ("configured", configured), ("best", tau)):
            results[f"{group}_f1@{label}"] = f1_at(*probs[group], threshold)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive the stop classification threshold.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with fold_<i>.pt checkpoints.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--travel-times-parquet", required=True)
    parser.add_argument("--dwell-times-parquet", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_travel, df_dwell = load_observations(args.travel_times_parquet, args.dwell_times_parquet)

    rows: List[Dict[str, float]] = []
    for fold in range(N_FOLDS):
        path = Path(args.checkpoint_dir) / f"fold_{fold}.pt"
        if not path.exists():
            logger.warning("Missing %s", path)
            continue
        rows.append(evaluate_fold(path, df_travel, df_dwell, args.artifacts_dir, device))
        logger.info("Fold %d: best threshold %.3f", fold, rows[-1]["best_threshold"])

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMean best threshold over {len(df)} folds: {df['best_threshold'].mean():.3f}")


if __name__ == "__main__":
    main()
