"""Data loading utilities for travel time and dwell time datasets."""
from pathlib import Path
from typing import Union, List

import pandas as pd

from time2model.logging import get_logger

logger = get_logger(__name__)


def load_travel_times(
    path: Union[str, Path, List[Union[str, Path]]],
    route: str | None = None,
) -> pd.DataFrame:
    """Load travel times from CSV, computing duration and link IDs."""
    # Handle single path or list of paths
    paths = [path] if isinstance(path, (str, Path)) else path
    
    if not paths:
        raise ValueError("At least one path must be provided")
    
    dfs = []
    for p in paths:
        columns = ["from_stop", "to_stop", "from_time", "to_time"]
        if route is not None:
            columns.append("route")
        
        df = pd.read_csv(
            p,
            usecols=columns,
            engine="pyarrow",
            dtype_backend="pyarrow",
        )
        
        if route is not None:
            route_filter = int(route) if route.isdigit() else route
            original_count = len(df)
            df = df[df["route"] == route_filter]
            logger.debug(
                "Filtered travel times to route '%s': %d -> %d records",
                route, original_count, len(df)
            )
            if len(df) == 0:
                logger.warning("No travel time records found for route '%s' in '%s'", route, p)
        
        df = df.astype({"from_stop": str, "to_stop": str})
        df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
        df["to_time"] = pd.to_datetime(df["to_time"], utc=True)
        df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()
        df["link"] = "link_" + df["from_stop"] + "_" + df["to_stop"]
        
        logger.debug("Loaded %d travel time records from '%s'", len(df), p)
        dfs.append(df)
    
    if len(dfs) == 1:
        result = dfs[0]
    else:
        result = pd.concat(dfs, ignore_index=True)
        logger.info("Combined %d travel time records from %d regions", len(result), len(dfs))
    
    return result


def load_dwell_times(
    path: Union[str, Path, List[Union[str, Path]]],
    route: str | None = None,
) -> pd.DataFrame:
    """Load dwell times from CSV, computing duration."""
    # Handle single path or list of paths
    paths = [path] if isinstance(path, (str, Path)) else path
    
    if not paths:
        raise ValueError("At least one path must be provided")
    
    dfs = []
    for p in paths:
        columns = ["stop", "from_time", "to_time"]
        if route is not None:
            columns.append("route")
        
        df = pd.read_csv(
            p,
            usecols=columns,
            engine="pyarrow",
            dtype_backend="pyarrow",
        )
        
        if route is not None:
            # Route column may be float in dwell times data
            route_filter = float(route) if route.replace('.', '').isdigit() else route
            original_count = len(df)
            df = df[df["route"] == route_filter]
            logger.debug(
                "Filtered dwell times to route '%s': %d -> %d records",
                route, original_count, len(df)
            )
            if len(df) == 0:
                logger.warning("No dwell time records found for route '%s' in '%s'", route, p)
        
        df = df.astype({"stop": str})
        df["from_time"] = pd.to_datetime(df["from_time"], utc=True)
        df["to_time"] = pd.to_datetime(df["to_time"], utc=True)
        df["travel_time"] = (df["to_time"] - df["from_time"]).dt.total_seconds()
        df["link"] = df["stop"]  # compatibility with FixedIntervalDataset
        
        logger.debug("Loaded %d dwell time records from '%s'", len(df), p)
        dfs.append(df)
    
    if len(dfs) == 1:
        result = dfs[0]
    else:
        result = pd.concat(dfs, ignore_index=True)
        logger.info("Combined %d dwell time records from %d regions", len(result), len(dfs))
    
    return result



def get_time_range(
    df_travel: pd.DataFrame,
    df_dwell: pd.DataFrame | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute overall time range across travel and dwell time datasets."""
    if df_dwell is not None:
        overall_start = min(df_travel["from_time"].min(), df_dwell["from_time"].min())
        overall_end = max(df_travel["from_time"].max(), df_dwell["from_time"].max())
    else:
        overall_start = df_travel["from_time"].min()
        overall_end = df_travel["from_time"].max()
    return overall_start, overall_end
