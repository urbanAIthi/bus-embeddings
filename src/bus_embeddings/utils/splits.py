"""Spatial (municipality) and temporal (expanding window) data splits."""
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from bus_embeddings.logging import get_logger
from bus_embeddings.training.config import MUNICIPALITY_CATEGORIES, N_FOLDS, REGION_SPLIT, VAL_FRACTION

logger = get_logger(__name__)

FoldBoundaries = Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]


def discover_regions(artifacts_dir: str | Path, municipalities: Iterable[str]) -> List[str]:
    """Municipalities that have both graph artifacts and observations."""
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.is_dir():
        raise ValueError(f"Artifacts directory does not exist: {artifacts_dir}")

    artifact_regions = {
        child.name
        for child in artifacts_path.iterdir()
        if child.is_dir()
        and all((child / f).exists() for f in ("nodes_link.csv", "nodes_stop.csv", "edges.csv"))
    }
    available = sorted(artifact_regions & set(municipalities))
    if not available:
        raise ValueError(
            f"No municipality has both graph artifacts {sorted(artifact_regions)} "
            f"and observations {sorted(municipalities)}"
        )
    logger.info("Found %d municipalities: %s", len(available), available)
    return available


def assign_regions_to_splits(available_regions: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Return the training, validation and test municipalities of ``REGION_SPLIT``."""
    available_set = set(available_regions)
    category_of = {m: c for c, members in MUNICIPALITY_CATEGORIES.items() for m in members}
    for split_name, regions in REGION_SPLIT.items():
        missing = [r for r in regions if r not in available_set]
        if missing:
            raise ValueError(f"{split_name} municipalities {missing} are not available")
        categories = [category_of[r] for r in regions]
        logger.info(
            "%s regions: %s (%s)", split_name, sorted(regions),
            ", ".join(f"{categories.count(c)} {c}" for c in MUNICIPALITY_CATEGORIES),
        )
    return (
        sorted(REGION_SPLIT["train"]),
        sorted(REGION_SPLIT["val"]),
        sorted(REGION_SPLIT["test"]),
    )


def region_artifact_paths(
    regions: List[str],
    artifacts_dir: str | Path,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Paths of the link node, stop node and edge CSVs of each municipality."""
    root = Path(artifacts_dir)
    return (
        [root / r / "nodes_link.csv" for r in regions],
        [root / r / "nodes_stop.csv" for r in regions],
        [root / r / "edges.csv" for r in regions],
    )


def generate_split_boundaries(
    start_time: pd.Timestamp, end_time: pd.Timestamp, val_size: float, n_splits: int
) -> List[FoldBoundaries]:
    """Boundaries ``(train_start, train_end, val_end, test_end)`` of each fold.

    The time range is divided into ``n_splits + 1`` segments. Fold ``i``
    trains on the first ``i + 1`` segments (expanding window); the next
    segment is split into a validation (``val_size``) and a test period.
    """
    split_size = (end_time - start_time) / (n_splits + 1)
    val_size_absolute = split_size * val_size
    test_size_absolute = split_size * (1 - val_size)

    splits = []
    for i in range(n_splits):
        train_end = start_time + (i + 1) * split_size
        val_end = train_end + val_size_absolute
        test_end = val_end + test_size_absolute
        splits.append((start_time, train_end, val_end, test_end))
    return splits


def fold_boundaries(df_travel: pd.DataFrame, df_dwell: pd.DataFrame) -> List[FoldBoundaries]:
    """Fold boundaries over the time range of the given observations."""
    start = min(df_travel["from_time"].min(), df_dwell["from_time"].min())
    end = max(df_travel["from_time"].max(), df_dwell["from_time"].max())
    logger.info("Time range: %s to %s", start, end)
    return generate_split_boundaries(start, end, val_size=VAL_FRACTION, n_splits=N_FOLDS)
