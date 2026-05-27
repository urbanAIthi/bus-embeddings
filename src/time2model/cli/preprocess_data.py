"""Preprocess raw CSV data into cleaned Parquet files for fast training."""
import argparse
from pathlib import Path

import pandas as pd

from time2model.logging import get_logger, setup_logging
from time2model.data.loaders import load_travel_times, load_dwell_times
from time2model.data.thresholds import (
    filter_travel_time_freeflow_ratio,
    filter_dwell_times,
)

logger = get_logger(__name__)


def _discover_csv_regions(
    travel_times_dir: str,
    dwell_times_dir: str,
) -> list[str]:
    """Return sorted municipality IDs that have both travel and dwell CSVs."""
    travel_path = Path(travel_times_dir)
    dwell_path = Path(dwell_times_dir)

    if not travel_path.is_dir():
        raise ValueError(f"Travel times directory does not exist: {travel_times_dir}")
    if not dwell_path.is_dir():
        raise ValueError(f"Dwell times directory does not exist: {dwell_times_dir}")

    travel_regions = {p.stem for p in travel_path.glob("*.csv")}
    dwell_regions = {p.stem for p in dwell_path.glob("*.csv")}
    available = sorted(travel_regions & dwell_regions)

    if not available:
        raise ValueError(
            f"No municipalities found with both travel and dwell time CSVs. "
            f"Travel: {sorted(travel_regions)}, Dwell: {sorted(dwell_regions)}"
        )
    return available


def _load_link_nodes(artifacts_dir: Path, region: str) -> pd.DataFrame:
    path = artifacts_dir / region / "nodes_link.csv"
    if not path.exists():
        raise ValueError(f"Missing link node artifacts for region {region}: {path}")
    return pd.read_csv(path, usecols=["id", "osm_length", "osm_travel_time"])


def _load_stop_nodes(artifacts_dir: Path, region: str) -> pd.DataFrame:
    path = artifacts_dir / region / "nodes_stop.csv"
    if not path.exists():
        raise ValueError(f"Missing stop node artifacts for region {region}: {path}")
    return pd.read_csv(path, usecols=["stop_id", "gtfs_is_terminus"])


def _drop_self_loops(df_tt: pd.DataFrame, region: str) -> pd.DataFrame:
    """Remove from_stop == to_stop rows (dwell records leaking into travel stream)."""
    mask = df_tt["from_stop"].astype(str) == df_tt["to_stop"].astype(str)
    n = int(mask.sum())
    if n > 0:
        logger.info(
            "Region %s: dropped %d self-loop travel-time rows (%.2f%%)",
            region, n, 100.0 * n / len(df_tt),
        )
        df_tt = df_tt[~mask].reset_index(drop=True)
    return df_tt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw CSV travel/dwell data into cleaned Parquet files."
    )
    parser.add_argument(
        "--travel-times-dir",
        type=str,
        required=True,
        help="Directory containing {municipality_id}.csv travel time files.",
    )
    parser.add_argument(
        "--dwell-times-dir",
        type=str,
        required=True,
        help="Directory containing {municipality_id}.csv dwell time files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Directory containing per-municipality graph artifacts "
             "(each with nodes_link.csv and nodes_stop.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    regions = _discover_csv_regions(args.travel_times_dir, args.dwell_times_dir)
    logger.info("Discovered %d municipalities: %s", len(regions), regions)

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    travel_dfs: list[pd.DataFrame] = []
    dwell_dfs: list[pd.DataFrame] = []

    for region in regions:
        tt_path = str(Path(args.travel_times_dir) / f"{region}.csv")
        dw_path = str(Path(args.dwell_times_dir) / f"{region}.csv")

        df_tt = load_travel_times(tt_path)
        df_dw = load_dwell_times(dw_path)

        n_tt_before = len(df_tt)
        n_dw_before = len(df_dw)

        link_nodes = _load_link_nodes(artifacts_dir, region)
        stop_nodes = _load_stop_nodes(artifacts_dir, region)

        df_tt = _drop_self_loops(df_tt, region)
        df_tt = filter_travel_time_freeflow_ratio(df_tt, link_nodes)
        df_dw = filter_dwell_times(df_dw, stop_nodes)

        logger.info(
            "Region %s: travel %d → %d (−%.1f%%), dwell %d → %d (−%.1f%%)",
            region,
            n_tt_before, len(df_tt),
            100.0 * (n_tt_before - len(df_tt)) / max(n_tt_before, 1),
            n_dw_before, len(df_dw),
            100.0 * (n_dw_before - len(df_dw)) / max(n_dw_before, 1),
        )

        df_tt["municipality"] = region
        df_dw["municipality"] = region

        travel_dfs.append(df_tt)
        dwell_dfs.append(df_dw)

    df_travel_all = pd.concat(travel_dfs, ignore_index=True)
    df_dwell_all = pd.concat(dwell_dfs, ignore_index=True)

    logger.info(
        "Combined: %d travel time records, %d dwell time records",
        len(df_travel_all), len(df_dwell_all),
    )

    for col in df_travel_all.select_dtypes(include=["string[pyarrow]"]).columns:
        df_travel_all[col] = df_travel_all[col].astype(str)
    for col in df_dwell_all.select_dtypes(include=["string[pyarrow]"]).columns:
        df_dwell_all[col] = df_dwell_all[col].astype(str)

    travel_out = output_dir / "travel_times.parquet"
    dwell_out = output_dir / "dwell_times.parquet"

    df_travel_all.to_parquet(travel_out, engine="pyarrow", index=False)
    df_dwell_all.to_parquet(dwell_out, engine="pyarrow", index=False)

    logger.info("Saved travel times → %s (%d rows)", travel_out, len(df_travel_all))
    logger.info("Saved dwell times  → %s (%d rows)", dwell_out, len(df_dwell_all))
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
