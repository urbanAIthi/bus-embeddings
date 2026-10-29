"""Data loading utilities for travel time and dwell time datasets."""
from pathlib import Path
from typing import Union

import pandas as pd

from bus_embeddings.logging import get_logger

logger = get_logger(__name__)


def load_travel_times(path: Union[str, Path]) -> pd.DataFrame:
    """Load travel times from CSV, computing duration and link IDs."""
    df = pd.read_csv(
        path,
        usecols=["from_stop", "to_stop", "from_time", "to_time"],
        engine="pyarrow",
        dtype_backend="pyarrow",
    )

    df = df.astype({"from_stop": str, "to_stop": str})
    df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
    df["to_time"] = pd.to_datetime(df["to_time"], utc=True)
    df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()
    df["link"] = "link_" + df["from_stop"] + "_" + df["to_stop"]

    logger.debug("Loaded %d travel time records from '%s'", len(df), path)
    return df


def load_dwell_times(path: Union[str, Path]) -> pd.DataFrame:
    """Load dwell times from CSV, computing duration (in the ``travel_time`` column)."""
    df = pd.read_csv(
        path,
        usecols=["stop", "from_time", "to_time"],
        engine="pyarrow",
        dtype_backend="pyarrow",
    )

    df = df.astype({"stop": str})
    df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
    df["to_time"] = pd.to_datetime(df["to_time"], utc=True)
    df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()

    logger.debug("Loaded %d dwell time records from '%s'", len(df), path)
    return df

