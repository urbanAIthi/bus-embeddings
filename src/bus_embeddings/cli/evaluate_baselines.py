"""Evaluate the rule-based travel time baselines.

Evaluates three baselines that predict a fixed travel time per link on the
same folds and subsets as the models of ``train_embeddings``:

    gtfs_schedule         scheduled travel time from the GTFS feed
    valhalla_speed_limit  ideal travel time from the posted speed limits
    valhalla_estimate     travel time estimate of Valhalla's bus costing

Writes ``metrics.json`` with the per-fold metrics and their aggregate over
folds to ``--output-dir``.

Example::

    python -m bus_embeddings.cli.evaluate_baselines \\
        --artifacts-dir data/artifacts \\
        --travel-times-parquet data/preprocessed/travel_times.parquet \\
        --dwell-times-parquet data/preprocessed/dwell_times.parquet \\
        --output-dir results/baselines
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from bus_embeddings.data.proxy_task_dataset import load_observations
from bus_embeddings.logging import get_logger, setup_logging
from bus_embeddings.training.metrics import compute_seasonal_naive_metrics
from bus_embeddings.utils.splits import assign_regions_to_splits, discover_regions, fold_boundaries

logger = get_logger(__name__)

# Baseline name -> column of nodes_link.csv
BASELINES = {
    "gtfs_schedule": "gtfs_scheduled_travel_time",
    "valhalla_speed_limit": "osm_travel_time",
    "valhalla_estimate": "osm_estimated_travel_time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the rule-based travel time baselines.")
    parser.add_argument(
        "--artifacts-dir", required=True,
        help="Directory with one sub-directory of graph artifacts per municipality.",
    )
    parser.add_argument("--travel-times-parquet", required=True, help="Preprocessed travel times.")
    parser.add_argument(
        "--dwell-times-parquet", required=True,
        help="Preprocessed dwell times (only used for the time range of the folds).",
    )
    parser.add_argument(
        "--output-dir", default="results/baselines",
        help="Directory for metrics.json (default: results/baselines).",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )
    return parser.parse_args()


def baseline_predictions(regions: List[str], artifacts_dir: str) -> Dict[str, Dict[str, float]]:
    """``{baseline: {link_id: predicted travel time}}`` for the links of ``regions``."""
    link_nodes = pd.concat(
        [pd.read_csv(Path(artifacts_dir) / r / "nodes_link.csv") for r in regions],
        ignore_index=True,
    )
    predictions = {}
    for name, column in BASELINES.items():
        mask = link_nodes[column].notna()
        predictions[name] = dict(zip(
            link_nodes.loc[mask, "id"].astype(str),
            link_nodes.loc[mask, column].astype(float),
        ))
    return predictions


def evaluate_baseline(
    df_eval: pd.DataFrame,
    predictions: Dict[str, float],
    df_history: pd.DataFrame,
) -> Dict[str, float]:
    """MAE, RMSE and seasonal MASE / RMSSE of a baseline on ``df_eval``."""
    preds = df_eval["link"].map(predictions)
    matched = preds.notna()
    if not matched.any():
        logger.warning("No observations matched the baseline predictions")
        return {}

    df_matched = df_eval.loc[matched]
    y_pred = torch.tensor(preds[matched].values, dtype=torch.float32).clamp(min=0.0)
    y_true = torch.tensor(df_matched["travel_time"].values, dtype=torch.float32)
    errors = y_pred - y_true
    metrics = {
        "mae": errors.abs().mean().item(),
        "rmse": (errors ** 2).mean().sqrt().item(),
    }
    scaled = compute_seasonal_naive_metrics(
        df_matched, df_history, "link", model_mae=metrics["mae"], model_rmse=metrics["rmse"],
    )
    for metric in ("mase", "rmsse"):
        if metric in scaled:
            metrics[metric] = scaled[metric]
    return metrics


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    df_travel_all, df_dwell_all = load_observations(
        args.travel_times_parquet, args.dwell_times_parquet,
    )
    municipalities = set(df_travel_all["municipality"].unique()) & set(df_dwell_all["municipality"].unique())
    train_regions, val_regions, test_regions = assign_regions_to_splits(
        discover_regions(args.artifacts_dir, municipalities)
    )
    regions = {"train": train_regions, "val": val_regions, "test": test_regions}
    travel = {g: df_travel_all[df_travel_all["municipality"].isin(r)] for g, r in regions.items()}
    dwell = {g: df_dwell_all[df_dwell_all["municipality"].isin(r)] for g, r in regions.items()}
    del df_travel_all, df_dwell_all

    predictions = {g: baseline_predictions(r, args.artifacts_dir) for g, r in regions.items()}
    boundaries = fold_boundaries(
        pd.concat([travel["train"], travel["val"], travel["test"]]),
        pd.concat([dwell["train"], dwell["val"], dwell["test"]]),
    )

    fold_metrics: List[Dict[str, float]] = []
    for fold, (train_start, train_end, val_end, test_end) in enumerate(boundaries):
        logger.info("FOLD %d/%d: train %s to %s, val to %s, test to %s",
                    fold + 1, len(boundaries), train_start, train_end, val_end, test_end)
        train, val, test = travel["train"], travel["val"], travel["test"]

        def window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
            return df[(df["from_time"] >= start) & (df["from_time"] < end)]

        # History for the seasonal naive forecast: all data before the end of the period
        val_history = pd.concat([train[train["from_time"] < val_end], val[val["from_time"] < val_end]], ignore_index=True)
        test_history = pd.concat([train[train["from_time"] < test_end], test[test["from_time"] < test_end]], ignore_index=True)
        subsets = {
            # subset: (observations, baseline predictions, history)
            "train": (window(train, train_start, train_end), predictions["train"], train[train["from_time"] < train_end]),
            "val_temporal": (window(train, train_end, val_end), predictions["train"], val_history),
            "val_spatial": (window(val, train_end, val_end), predictions["val"], val_history),
            "test_temporal": (window(train, val_end, test_end), predictions["train"], test_history),
            "test_spatial": (window(test, val_end, test_end), predictions["test"], test_history),
        }

        metrics: Dict[str, float] = {}
        for subset, (df_eval, subset_predictions, df_history) in subsets.items():
            if len(df_eval) == 0:
                logger.warning("  %s: no observations", subset)
                continue
            for name in BASELINES:
                for metric, value in evaluate_baseline(df_eval, subset_predictions[name], df_history).items():
                    metrics[f"{subset}/{name}/{metric}"] = value
                logger.info(
                    "  %-13s | %-20s | MASE %.4f | RMSSE %.4f", subset, name,
                    metrics.get(f"{subset}/{name}/mase", float("nan")),
                    metrics.get(f"{subset}/{name}/rmsse", float("nan")),
                )
        fold_metrics.append(metrics)

    # Geometric mean over folds for MASE / RMSSE, arithmetic mean otherwise
    aggregated: Dict[str, float] = {}
    for key in sorted({k for m in fold_metrics for k in m}):
        values = [m[key] for m in fold_metrics if key in m]
        if key.endswith(("/mase", "/rmsse")):
            aggregated[key] = float(np.exp(np.mean(np.log(values))))
        else:
            aggregated[key] = float(np.mean(values))
    for key, value in aggregated.items():
        logger.info("  %s: %.4f", key, value)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"regions": regions, "folds": fold_metrics, "aggregate": aggregated}, f, indent=2)
    logger.info("Saved %s", metrics_path)


if __name__ == "__main__":
    main()
