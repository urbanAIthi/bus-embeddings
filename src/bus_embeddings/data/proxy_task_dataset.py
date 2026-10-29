"""Proxy-task observations: one sample per observed travel or dwell time.

Each link observation yields a travel time target relative to the ideal
(free-flow) travel time of the link. Each stop observation yields a stop
label (dwell time > 0) and, if the bus stopped, a dwell time target. Targets
are z-scored with scalers fitted on the training data.
"""
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import holidays
import numpy as np
import pandas as pd
import torch

from bus_embeddings.logging import get_logger

logger = get_logger(__name__)

_NL_HOLIDAYS = holidays.Netherlands(years=range(2015, 2030))  # type: ignore[attr-defined]

# Time context: sin/cos time of day, day-of-week one-hot (7), sin/cos month
# of year, public holiday flag.
TIME_CONTEXT_DIM = 12

# One batch: (link_indices, link_ctx, link_targets,
#             stop_indices, stop_ctx, stop_targets, stop_labels)
Batch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    Optional[torch.Tensor],
]


def load_observations(
    travel_times_parquet: str | Path,
    dwell_times_parquet: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preprocessed travel and dwell time observations (UTC timestamps)."""
    df_travel = pd.read_parquet(travel_times_parquet, engine="pyarrow")
    df_dwell = pd.read_parquet(dwell_times_parquet, engine="pyarrow")
    for col in ["from_time", "to_time"]:
        if df_travel[col].dt.tz is None:
            df_travel[col] = df_travel[col].dt.tz_localize("UTC")
        if df_dwell[col].dt.tz is None:
            df_dwell[col] = df_dwell[col].dt.tz_localize("UTC")
    logger.info(
        "Loaded %d travel time and %d dwell time observations",
        len(df_travel), len(df_dwell),
    )
    return df_travel, df_dwell


class GlobalScaler:
    """Z-score scaler with a single global mean and standard deviation."""

    def __init__(self, eps: float = 1e-8):
        self.mean_: np.ndarray = np.array([0.0])
        self.scale_: np.ndarray = np.array([1.0])
        self.eps = eps
        self._fitted = False

    @property
    def mean(self) -> float:
        return float(self.mean_[0])

    @property
    def std(self) -> float:
        return float(self.scale_[0])

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, values: np.ndarray) -> "GlobalScaler":
        """Fit on all non-NaN values."""
        values = values.flatten()
        values = values[~np.isnan(values)]
        if len(values) == 0:
            logger.warning("GlobalScaler.fit() received no valid values, using defaults")
            self.mean_ = np.array([0.0])
            self.scale_ = np.array([1.0])
        else:
            self.mean_ = np.array([float(np.mean(values))])
            std = float(np.std(values))
            self.scale_ = np.array([std if std > self.eps else self.eps])
        self._fitted = True
        return self

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


def compute_time_context(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Time context of each timestamp, ``[n, TIME_CONTEXT_DIM]``."""
    minutes = timestamps.hour * 60 + timestamps.minute
    tod_fraction = minutes / 1440.0
    sin_tod = np.sin(2 * np.pi * tod_fraction)
    cos_tod = np.cos(2 * np.pi * tod_fraction)

    dow = timestamps.dayofweek.values  # 0=Monday, 6=Sunday
    dow_onehot = np.zeros((len(timestamps), 7), dtype=np.float32)
    dow_onehot[np.arange(len(timestamps)), dow] = 1.0

    month_fraction = (timestamps.month - 1) / 12.0
    sin_month = np.sin(2 * np.pi * month_fraction)
    cos_month = np.cos(2 * np.pi * month_fraction)

    # Look up holidays once per day
    day_keys = timestamps.normalize()
    unique_days = day_keys.unique()
    holiday_lookup = pd.Series(
        [1.0 if d.date() in _NL_HOLIDAYS else 0.0 for d in unique_days],
        index=unique_days,
        dtype=np.float32,
    )
    is_holiday = day_keys.map(holiday_lookup).to_numpy(dtype=np.float32)

    time_context = np.column_stack(
        [sin_tod, cos_tod, dow_onehot, sin_month, cos_month, is_holiday]
    )
    return time_context.astype(np.float32)


class ProxyTaskBatchIterator:
    """Batch iterator over the link and stop samples of a dataset.

    Yields :data:`Batch` tuples by slicing the dataset tensors. With
    ``subsample < 1.0`` each pass draws a fresh random subset of that
    fraction of the samples (requires ``shuffle=True``).
    """

    def __init__(
        self,
        dataset: "RawObservationProxyDataset",
        batch_size: int,
        shuffle: bool = False,
        pin_memory: bool = False,
        subsample: float = 1.0,
    ) -> None:
        if not (0.0 < subsample <= 1.0):
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if subsample < 1.0 and not shuffle:
            raise ValueError("subsample < 1.0 requires shuffle=True")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self._n_link = dataset.num_link_samples
        self._n_stop = dataset.num_stop_samples
        self._n_total = self._n_link + self._n_stop
        self._n_used = int(self._n_total * subsample)

    def __len__(self) -> int:
        return (self._n_used + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        if self.shuffle:
            perm = torch.randperm(self._n_total)[: self._n_used]
        else:
            perm = torch.arange(self._n_total)

        ds = self.dataset
        ctx_dim = ds._link_time_context.shape[1]

        for start in range(0, self._n_used, self.batch_size):
            batch_idx = perm[start : start + self.batch_size]

            link_mask = batch_idx < self._n_link
            link_sel = batch_idx[link_mask]
            stop_sel = batch_idx[~link_mask] - self._n_link

            if link_sel.numel() > 0:
                link_indices = ds._link_node_indices[link_sel]
                link_ctx = ds._link_time_context[link_sel]
                link_targets = ds._link_targets[link_sel]
            else:
                link_indices = torch.empty(0, dtype=torch.long)
                link_ctx = torch.empty(0, ctx_dim, dtype=torch.float32)
                link_targets = torch.empty(0, dtype=torch.float32)

            stop_labels: Optional[torch.Tensor]
            if stop_sel.numel() > 0:
                stop_indices = ds._stop_node_indices[stop_sel]
                stop_ctx = ds._stop_time_context[stop_sel]
                stop_targets = ds._stop_targets[stop_sel]
                stop_labels = ds._stop_labels[stop_sel]
            else:
                stop_indices = torch.empty(0, dtype=torch.long)
                stop_ctx = torch.empty(0, ctx_dim, dtype=torch.float32)
                stop_targets = torch.empty(0, dtype=torch.float32)
                stop_labels = None

            if self.pin_memory:
                link_indices = link_indices.pin_memory()
                link_ctx = link_ctx.pin_memory()
                link_targets = link_targets.pin_memory()
                stop_indices = stop_indices.pin_memory()
                stop_ctx = stop_ctx.pin_memory()
                stop_targets = stop_targets.pin_memory()
                if stop_labels is not None:
                    stop_labels = stop_labels.pin_memory()

            yield (
                link_indices, link_ctx, link_targets,
                stop_indices, stop_ctx, stop_targets,
                stop_labels,
            )


class RawObservationProxyDataset:
    """Proxy-task samples built from individual travel and dwell time observations.

    Args:
        df_travel: Travel time observations (``link``, ``from_time``, ``travel_time``).
        df_dwell: Dwell time observations (``stop``, ``from_time``, ``travel_time``).
        link_nodes: Link nodes of the graph; their row order defines the node indices.
        stop_nodes: Stop nodes of the graph; their row order defines the node indices.
        freeflow_times: Ideal travel time per link ID (seconds).
        freeflow_ratio: Link target is ``travel_time / ideal`` (True) or
            ``travel_time - ideal`` (False).
        log_transform: Model travel and dwell times in log1p space (difference
            target only).
        link_scaler, stop_scaler: Fitted target scalers; fitted on this
            dataset if ``None`` (the dwell scaler on stopped observations only).
    """

    def __init__(
        self,
        df_travel: pd.DataFrame,
        df_dwell: pd.DataFrame,
        link_nodes: pd.DataFrame,
        stop_nodes: pd.DataFrame,
        freeflow_times: Dict[str, float],
        freeflow_ratio: bool,
        log_transform: bool = False,
        link_scaler: Optional[GlobalScaler] = None,
        stop_scaler: Optional[GlobalScaler] = None,
    ):
        if freeflow_ratio and log_transform:
            raise ValueError("The travel time ratio target cannot be combined with log_transform")

        missing_travel = {"link", "from_time", "travel_time"} - set(df_travel.columns)
        if missing_travel:
            raise ValueError(f"df_travel missing required columns: {missing_travel}")
        missing_dwell = {"stop", "from_time", "travel_time"} - set(df_dwell.columns)
        if missing_dwell:
            raise ValueError(f"df_dwell missing required columns: {missing_dwell}")

        self.link_scaler = link_scaler if link_scaler is not None else GlobalScaler()
        self.stop_scaler = stop_scaler if stop_scaler is not None else GlobalScaler()

        link_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(link_nodes["id"])}
        stop_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(stop_nodes["id"])}

        # --- Link (travel time) samples ---
        df_travel = df_travel.copy()
        df_travel["link_str"] = df_travel["link"].astype(str)
        df_travel_valid = df_travel[df_travel["link_str"].isin(link_id_to_idx)]
        if len(df_travel_valid) < len(df_travel):
            logger.debug("Dropped %d travel times of links not in the graph",
                         len(df_travel) - len(df_travel_valid))
        if len(df_travel_valid) == 0:
            raise ValueError("No travel time observations match the link nodes of the graph")

        link_travel_times = df_travel_valid["travel_time"].to_numpy().copy()
        ff_values = df_travel_valid["link_str"].map(freeflow_times).to_numpy(dtype=np.float64)
        missing_ff = np.isnan(ff_values)
        if missing_ff.any():
            missing_ids = df_travel_valid.loc[missing_ff, "link_str"].unique()  # type: ignore[union-attr]
            raise ValueError(
                f"Missing ideal travel times for {missing_ff.sum()} observations "
                f"({len(missing_ids)} links), e.g. {list(missing_ids[:5])}"
            )
        # Per-observation reference used to convert predictions back to seconds
        if freeflow_ratio:
            ff_safe = np.maximum(ff_values, 1.0)  # avoid division by zero
            link_values = link_travel_times / ff_safe
            self._link_freeflow_refs = torch.from_numpy(ff_safe.astype(np.float32))
        elif log_transform:
            link_values = np.log1p(link_travel_times) - np.log1p(ff_values)
            self._link_freeflow_refs = torch.from_numpy(np.log1p(ff_values).astype(np.float32))
        else:
            link_values = link_travel_times - ff_values
            self._link_freeflow_refs = torch.from_numpy(ff_values.astype(np.float32))

        if not self.link_scaler.is_fitted:
            self.link_scaler.fit(link_values)

        link_node_indices = df_travel_valid["link_str"].map(link_id_to_idx).values
        self._link_node_indices = torch.from_numpy(link_node_indices.astype(np.int64))
        self._link_municipalities: Optional[np.ndarray] = (
            df_travel_valid["municipality"].values
            if "municipality" in df_travel_valid.columns else None
        )
        normalized_link_times = (link_values - self.link_scaler.mean) / self.link_scaler.std
        self._link_targets = torch.from_numpy(normalized_link_times.astype(np.float32))
        self._link_time_context = torch.from_numpy(
            compute_time_context(pd.DatetimeIndex(df_travel_valid["from_time"]))
        )
        self._num_link_samples = len(df_travel_valid)

        # --- Stop (dwell time) samples ---
        df_dwell = df_dwell.copy()
        df_dwell["stop_str"] = "stop_" + df_dwell["stop"].astype(str)
        df_dwell_valid = df_dwell[df_dwell["stop_str"].isin(stop_id_to_idx)]
        if len(df_dwell_valid) < len(df_dwell):
            logger.debug("Dropped %d dwell times of stops not in the graph",
                         len(df_dwell) - len(df_dwell_valid))
        if len(df_dwell_valid) == 0:
            raise ValueError("No dwell time observations match the stop nodes of the graph")

        dwell_times = df_dwell_valid["travel_time"].to_numpy()  # dwell time in seconds
        stopped_mask = dwell_times > 0
        if log_transform:
            dwell_times = np.log1p(dwell_times)

        if not self.stop_scaler.is_fitted:
            stopped_dwell_values = dwell_times[stopped_mask]
            if len(stopped_dwell_values) == 0:
                raise ValueError("No stopped observations (dwell > 0) to fit the dwell time scaler")
            self.stop_scaler.fit(stopped_dwell_values)

        stop_node_indices = df_dwell_valid["stop_str"].map(stop_id_to_idx).values
        self._stop_node_indices = torch.from_numpy(stop_node_indices.astype(np.int64))
        normalized_dwell_times = (dwell_times - self.stop_scaler.mean) / self.stop_scaler.std
        self._stop_targets = torch.from_numpy(normalized_dwell_times.astype(np.float32))
        self._stop_labels = torch.from_numpy(stopped_mask.astype(np.float32))
        # Aligned with the stopped observations (the dwell time regression targets)
        self._stop_municipalities: Optional[np.ndarray] = (
            df_dwell_valid["municipality"].values[stopped_mask]
            if "municipality" in df_dwell_valid.columns else None
        )
        self._stop_time_context = torch.from_numpy(
            compute_time_context(pd.DatetimeIndex(df_dwell_valid["from_time"]))
        )
        self._num_stop_samples = len(df_dwell_valid)

        logger.info("Created %d link samples and %d stop samples",
                    self._num_link_samples, self._num_stop_samples)

    def __len__(self) -> int:
        return self._num_link_samples + self._num_stop_samples

    @property
    def num_link_samples(self) -> int:
        return self._num_link_samples

    @property
    def num_stop_samples(self) -> int:
        return self._num_stop_samples

    @property
    def link_municipalities(self) -> Optional[np.ndarray]:
        """Municipality of each link observation."""
        return self._link_municipalities

    @property
    def stop_municipalities(self) -> Optional[np.ndarray]:
        """Municipality of each stopped dwell time observation."""
        return self._stop_municipalities

    @property
    def link_freeflow_refs(self) -> torch.Tensor:
        """Per-observation ideal travel time used to convert link predictions to seconds."""
        return self._link_freeflow_refs
