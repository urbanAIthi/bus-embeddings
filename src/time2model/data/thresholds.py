"""Outlier filtering for travel times and dwell times."""
import pandas as pd

from time2model.logging import get_logger
from time2model.training.config import (
    TRAVEL_TIME_FREEFLOW_RATIO_MIN,
    TRAVEL_TIME_FREEFLOW_RATIO_MAX,
    DWELL_TIME_CAP,
)

logger = get_logger(__name__)


def filter_travel_time_freeflow_ratio(
    df_travel: pd.DataFrame,
    link_nodes: pd.DataFrame,
    lower: float = TRAVEL_TIME_FREEFLOW_RATIO_MIN,
    upper: float = TRAVEL_TIME_FREEFLOW_RATIO_MAX,
) -> pd.DataFrame:
    """Drop observations whose travel_time / osm_travel_time is outside bounds.

    Keeps observations where ``lower <= travel_time / osm_travel_time <= upper``.
    Applied to *all* splits (train, val, test). The bounds are fixed (not
    fitted), so the filter is symmetric across splits and does not leak training
    distribution statistics into evaluation.

    Args:
        df_travel: Must contain ``link`` and ``travel_time`` columns.
        link_nodes: Must contain ``id`` and ``osm_travel_time`` columns.
        lower: Minimum allowed ratio.
        upper: Maximum allowed ratio.

    Returns:
        Filtered copy of *df_travel*.
    """
    freeflow = link_nodes.set_index("id")["osm_travel_time"]
    osm_tt = df_travel["link"].map(freeflow)

    valid_freeflow = osm_tt > 0
    ratio = df_travel["travel_time"] / osm_tt.where(valid_freeflow)

    mask = valid_freeflow & (ratio >= lower) & (ratio <= upper)
    n_removed = int((~mask).sum())
    if n_removed > 0:
        logger.info(
            "Freeflow-ratio filter [%.2f, %.2f]: removed %d / %d rows (%.2f%%)",
            lower, upper, n_removed, len(df_travel),
            100.0 * n_removed / len(df_travel),
        )
    return df_travel.loc[mask].reset_index(drop=True)


def filter_dwell_times(
    df_dwell: pd.DataFrame,
    stop_nodes: pd.DataFrame,
    cap: float = DWELL_TIME_CAP,
) -> pd.DataFrame:
    """Filter dwell time observations.

    - Drops all observations at terminus stops (``gtfs_is_terminus == 1``).
      These represent layover/recovery times — a different process from
      passenger-service dwell — with median ~80 s and p95 ~1200 s.
    - Caps non-terminus dwells at ``cap`` seconds (default 90 s). Above that
      the observations are almost exclusively recording artefacts or extended
      driver breaks mid-route, not true passenger dwell.

    Applied to *all* splits (train, val, test).

    Args:
        df_dwell: Must contain ``stop`` and ``travel_time`` columns.
        stop_nodes: Must contain ``stop_id`` and ``gtfs_is_terminus`` columns.
        cap: Maximum dwell time in seconds for non-terminus stops.

    Returns:
        Filtered copy of *df_dwell*.
    """
    n_before = len(df_dwell)

    terminus_ids = set(
        stop_nodes.loc[stop_nodes["gtfs_is_terminus"] == 1, "stop_id"].astype(str)
    )

    is_terminus = df_dwell["stop"].astype(str).isin(terminus_ids)
    n_terminus = int(is_terminus.sum())
    if n_terminus > 0:
        logger.info(
            "Dwell filter: dropping %d terminus-stop observations "
            "(%.2f%%, %d unique terminus stops)",
            n_terminus,
            100.0 * n_terminus / n_before,
            int(df_dwell.loc[is_terminus, "stop"].nunique()),
        )
        df_dwell = df_dwell[~is_terminus].reset_index(drop=True)

    over_cap = df_dwell["travel_time"] > cap
    n_over = int(over_cap.sum())
    if n_over > 0:
        logger.info(
            "Dwell filter: dropping %d non-terminus observations "
            "with dwell > %.0f s (%.2f%%)",
            n_over, cap,
            100.0 * n_over / len(df_dwell),
        )
        df_dwell = df_dwell[~over_cap].reset_index(drop=True)

    n_removed = n_before - len(df_dwell)
    if n_removed > 0:
        logger.info(
            "Dwell filter total: removed %d / %d rows (%.2f%%)",
            n_removed, n_before, 100.0 * n_removed / n_before,
        )
    return df_dwell
