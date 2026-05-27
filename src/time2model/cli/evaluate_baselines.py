"""Evaluate static baselines (Valhalla freeflow / estimated, GTFS scheduled)
against observed travel times, using the same region split and temporal CV
as the GNN model.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from time2model.utils.splits import generate_split_boundaries

from time2model.logging import get_logger, setup_logging
from time2model.training.config import N_SPLITS, SEED
from time2model.training.metrics import (
    compute_node_level_metrics,
    compute_seasonal_naive_metrics,
    compute_smape,
)
from time2model.training.train_utils import seed_all

# Reuse region-split logic from the training CLI.
from time2model.cli.train_embeddings import (
    assign_regions_to_splits,
    discover_regions,
)
from time2model.data.loaders import get_time_range

logger = get_logger(__name__)


def _build_baseline_predictions(
    link_nodes: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Return ``{baseline_name: {link_id: predicted_travel_time}}``."""
    baselines: Dict[str, Dict[str, float]] = {}

    if "osm_travel_time" in link_nodes.columns:
        mask = link_nodes["osm_travel_time"].notna()
        baselines["valhalla_freeflow"] = dict(zip(
            link_nodes.loc[mask, "id"].astype(str),
            link_nodes.loc[mask, "osm_travel_time"].astype(float),
        ))

    if "osm_estimated_travel_time" in link_nodes.columns:
        mask = link_nodes["osm_estimated_travel_time"].notna()
        baselines["valhalla_estimated"] = dict(zip(
            link_nodes.loc[mask, "id"].astype(str),
            link_nodes.loc[mask, "osm_estimated_travel_time"].astype(float),
        ))

    if "gtfs_scheduled_travel_time" in link_nodes.columns:
        mask = link_nodes["gtfs_scheduled_travel_time"].notna()
        baselines["gtfs_scheduled"] = dict(zip(
            link_nodes.loc[mask, "id"].astype(str),
            link_nodes.loc[mask, "gtfs_scheduled_travel_time"].astype(float),
        ))

    if not baselines:
        raise ValueError(
            "No baseline columns found in nodes_link.csv. "
            "Expected at least one of: osm_travel_time, "
            "osm_estimated_travel_time, gtfs_scheduled_travel_time."
        )

    return baselines


def _evaluate_baseline(
    df_obs: pd.DataFrame,
    pred_dict: Dict[str, float],
    df_history: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute all metrics for a single baseline on a set of observations."""
    preds = df_obs["link"].map(pred_dict)
    matched = preds.notna()
    if matched.sum() == 0:
        logger.warning("No observations matched baseline predictions.")
        return {}

    n_total = len(df_obs)
    n_matched = int(matched.sum())
    df_matched = df_obs.loc[matched].copy()
    y_pred = torch.tensor(preds[matched].values, dtype=torch.float32).clamp(min=0.0)
    y_true = torch.tensor(df_matched["travel_time"].values, dtype=torch.float32)

    metrics: Dict[str, float] = {}
    metrics["n_observations"] = n_total
    metrics["n_matched"] = n_matched
    metrics["coverage"] = n_matched / n_total if n_total > 0 else 0.0

    errors = y_pred - y_true
    metrics["mae"] = errors.abs().mean().item()
    metrics["rmse"] = (errors ** 2).mean().sqrt().item()
    metrics["bias"] = errors.mean().item()
    metrics["smape"] = compute_smape(y_pred, y_true)

    # Reconstruct a 12-dim time context (sin/cos tod + dow one-hot) so we can
    # reuse compute_node_level_metrics; remaining columns stay zero.
    from_time = pd.to_datetime(df_matched["from_time"])
    hours = from_time.dt.hour.values
    dow = from_time.dt.dayofweek.values

    n = len(df_matched)
    ctx = np.zeros((n, 12), dtype=np.float32)
    hour_frac = hours / 24.0
    ctx[:, 0] = np.sin(2 * np.pi * hour_frac)
    ctx[:, 1] = np.cos(2 * np.pi * hour_frac)
    for i in range(7):
        ctx[:, 2 + i] = (dow == i).astype(np.float32)
    time_ctx = torch.tensor(ctx)

    link_ids = df_matched["link"].values
    unique_links = np.unique(link_ids)
    link_to_idx = {lid: idx for idx, lid in enumerate(unique_links)}
    node_indices = torch.tensor(
        np.array([link_to_idx[lid] for lid in link_ids], dtype=np.int64)
    )

    node_metrics = compute_node_level_metrics(y_pred, y_true, node_indices, time_ctx)
    for k, v in node_metrics.items():
        metrics[k] = v

    if df_history is not None and len(df_history) > 0:
        sn = compute_seasonal_naive_metrics(
            df_eval=df_matched,
            df_history=df_history,
            id_col="link",
            time_col="from_time",
            value_col="travel_time",
            model_mae=metrics["mae"],
            model_rmse=metrics["rmse"],
        )
        for k, v in sn.items():
            metrics[k] = v

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate static baselines for travel time prediction."
    )
    parser.add_argument(
        "--artifacts-dir", type=str, required=True,
        help="Directory containing per-municipality artifact subdirectories.",
    )
    parser.add_argument(
        "--travel-times-parquet", type=str, required=True,
        help="Path to preprocessed travel_times.parquet.",
    )
    parser.add_argument(
        "--dwell-times-parquet", type=str, required=True,
        help="Path to preprocessed dwell_times.parquet (needed for time range).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for region assignment (default: {SEED}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results JSON. Defaults to artifacts-dir.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _load_link_nodes(regions: List[str], artifacts_dir: str) -> pd.DataFrame:
    """Load and concatenate nodes_link.csv for a list of municipalities."""
    dfs = []
    for r in regions:
        path = Path(artifacts_dir) / r / "nodes_link.csv"
        df = pd.read_csv(path)
        df["municipality"] = r
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _format_table(
    results: Dict[str, Dict[str, float]],
    split_label: str,
) -> str:
    """Format a comparison table for terminal output."""
    if not results:
        return f"  {split_label}: no results\n"

    header_metrics = ["mae", "rmse", "bias", "smape", "node_r2", "node_mape",
                      "node_mae", "node_smape", "smase", "srmsse", "coverage"]
    col_fmt = {
        "mae": ".2f", "rmse": ".2f", "bias": ".2f", "smape": ".1f",
        "node_r2": ".4f", "node_mape": ".1f", "node_mae": ".2f",
        "node_smape": ".1f", "smase": ".4f", "srmsse": ".4f", "coverage": ".1%",
    }

    # Header
    name_w = max(len(b) for b in results) + 2
    lines = [f"\n  {split_label}"]
    header = f"  {'Baseline':<{name_w}}"
    for m in header_metrics:
        header += f"  {m:>10}"
    lines.append(header)
    lines.append("  " + "─" * len(header))

    for bname, metrics in sorted(results.items()):
        row = f"  {bname:<{name_w}}"
        for m in header_metrics:
            val = metrics.get(m)
            if val is None:
                row += f"  {'N/A':>10}"
            else:
                row += f"  {val:>{10}{col_fmt.get(m, '.4f')}}"
        lines.append(row)

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    seed_all(args.seed)

    logger.info("Loading preprocessed parquet files...")
    df_travel_all = pd.read_parquet(args.travel_times_parquet, engine="pyarrow")
    df_dwell_all = pd.read_parquet(args.dwell_times_parquet, engine="pyarrow")

    for col in ["from_time", "to_time"]:
        if df_travel_all[col].dt.tz is None:
            df_travel_all[col] = df_travel_all[col].dt.tz_localize("UTC")
        if df_dwell_all[col].dt.tz is None:
            df_dwell_all[col] = df_dwell_all[col].dt.tz_localize("UTC")

    logger.info(
        "Loaded %d travel time records, %d dwell time records",
        len(df_travel_all), len(df_dwell_all),
    )

    # Same region split as the GNN training CLI.
    parquet_municipalities = (
        set(df_travel_all["municipality"].unique()) &
        set(df_dwell_all["municipality"].unique())
    )
    available_regions = discover_regions(args.artifacts_dir, parquet_municipalities)
    train_regions, val_regions, test_regions = assign_regions_to_splits(
        available_regions,
    )
    logger.info("Train: %s", train_regions)
    logger.info("Val:   %s", val_regions)
    logger.info("Test:  %s", test_regions)

    df_travel_train_full = df_travel_all[df_travel_all["municipality"].isin(train_regions)]
    df_travel_val_full = df_travel_all[df_travel_all["municipality"].isin(val_regions)]
    df_travel_test_full = df_travel_all[df_travel_all["municipality"].isin(test_regions)]
    del df_travel_all

    train_link_nodes = _load_link_nodes(train_regions, args.artifacts_dir)
    val_link_nodes = _load_link_nodes(val_regions, args.artifacts_dir)
    test_link_nodes = _load_link_nodes(test_regions, args.artifacts_dir)

    train_baselines = _build_baseline_predictions(train_link_nodes)
    val_baselines = _build_baseline_predictions(val_link_nodes)
    test_baselines = _build_baseline_predictions(test_link_nodes)
    logger.info("Baselines available: %s", sorted(val_baselines.keys()))

    overall_start, overall_end = get_time_range(
        pd.concat([df_travel_train_full, df_travel_val_full, df_travel_test_full],
                   ignore_index=True),
        df_dwell_all,
    )
    del df_dwell_all
    logger.info("Time range: %s to %s", overall_start, overall_end)

    split_boundaries = generate_split_boundaries(
        start_time=overall_start,
        end_time=overall_end,
        val_size=0.5,
        n_splits=N_SPLITS,
    )

    all_results: Dict[str, List[Dict[str, float]]] = {}

    for split_idx in range(N_SPLITS):
        train_start, train_end, val_end, test_end = split_boundaries[split_idx]
        val_start = train_end
        test_start = val_end

        logger.info("=" * 60)
        logger.info("SPLIT %d/%d", split_idx + 1, N_SPLITS)
        logger.info("  Train: %s → %s", train_start, train_end)
        logger.info("  Val:   %s → %s", val_start, val_end)
        logger.info("  Test:  %s → %s", test_start, test_end)

        # Slice observations by time window
        df_train = df_travel_train_full[
            (df_travel_train_full["from_time"] >= train_start) &
            (df_travel_train_full["from_time"] < train_end)
        ]
        df_val = df_travel_val_full[
            (df_travel_val_full["from_time"] >= val_start) &
            (df_travel_val_full["from_time"] < val_end)
        ]
        df_test = df_travel_test_full[
            (df_travel_test_full["from_time"] >= test_start) &
            (df_travel_test_full["from_time"] < test_end)
        ]
        # Train regions at val/test time windows
        df_train_valtime = df_travel_train_full[
            (df_travel_train_full["from_time"] >= val_start) &
            (df_travel_train_full["from_time"] < val_end)
        ]
        df_train_testtime = df_travel_train_full[
            (df_travel_train_full["from_time"] >= test_start) &
            (df_travel_train_full["from_time"] < test_end)
        ]

        # History for seasonal naive = all data up to evaluation start
        df_train_history = df_travel_train_full[
            df_travel_train_full["from_time"] < train_end
        ]
        df_val_history = pd.concat([
            df_travel_train_full[df_travel_train_full["from_time"] < val_end],
            df_travel_val_full[df_travel_val_full["from_time"] < val_end],
        ], ignore_index=True)
        df_test_history = pd.concat([
            df_travel_train_full[df_travel_train_full["from_time"] < test_end],
            df_travel_test_full[df_travel_test_full["from_time"] < test_end],
        ], ignore_index=True)

        for eval_label, df_eval, baselines, df_hist in [
            ("train", df_train, train_baselines, df_train_history),
            ("val_original", df_train_valtime, train_baselines, df_val_history),
            ("val_transfer", df_val, val_baselines, df_val_history),
            ("test_original", df_train_testtime, train_baselines, df_test_history),
            ("test_transfer", df_test, test_baselines, df_test_history),
        ]:
            if len(df_eval) == 0:
                logger.warning("  %s: no observations, skipping.", eval_label)
                continue

            for bname, pred_dict in baselines.items():
                key = f"{eval_label}/{bname}"
                metrics = _evaluate_baseline(df_eval, pred_dict, df_history=df_hist)
                if not metrics:
                    continue

                logger.info(
                    "  %s | %s: MAE=%.2f  RMSE=%.2f  sMAPE=%.1f  "
                    "node_R²=%.4f  node_MAPE=%.1f  sMASE=%s  sRMSSE=%s",
                    eval_label, bname,
                    metrics["mae"], metrics["rmse"], metrics["smape"],
                    metrics.get("node_r2", 0),
                    metrics.get("node_mape", 0),
                    f"{metrics['smase']:.4f}" if "smase" in metrics else "N/A",
                    f"{metrics['srmsse']:.4f}" if "srmsse" in metrics else "N/A",
                )

                all_results.setdefault(key, []).append(metrics)

    agg_metrics = [
        "mae", "rmse", "bias", "smape",
        "node_r2", "node_mape", "node_mae", "node_smape",
        "smase", "srmsse", "coverage",
    ]
    # Match GNN evaluation: geometric mean for scale-relative metrics.
    geomean_metrics = {"smase", "srmsse"}

    summary: Dict[str, Dict[str, float]] = {}
    for key, split_list in all_results.items():
        agg: Dict[str, float] = {}
        for m in agg_metrics:
            vals = [s[m] for s in split_list if m in s]
            if vals:
                if m in geomean_metrics:
                    agg[m] = float(np.exp(np.mean(np.log(vals))))
                else:
                    agg[m] = float(np.mean(vals))
        agg["n_splits"] = len(split_list)
        summary[key] = agg

    print("\n" + "=" * 80)
    print("BASELINE EVALUATION SUMMARY (averaged across splits)")
    print("=" * 80)

    # Group by eval_label
    for eval_label in ["train", "val_original", "val_transfer", "test_original", "test_transfer"]:
        group = {
            k.split("/", 1)[1]: v
            for k, v in summary.items()
            if k.startswith(eval_label + "/")
        }
        if group:
            print(_format_table(group, eval_label))

    print()

    output_dir = Path(args.output_dir or args.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_results.json"

    # Convert numpy types for JSON serialization
    def _to_python(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    serializable = {
        k: {mk: _to_python(mv) for mk, mv in v.items()}
        for k, v in summary.items()
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
