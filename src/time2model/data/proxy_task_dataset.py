"""Dataset for proxy-task training: (node, timestamp) samples with time
context, used to learn GNN embeddings by predicting travel / dwell times
from static graph features.
"""
from typing import Tuple, Optional

import holidays
import numpy as np
import pandas as pd
import torch

from time2model.logging import get_logger

_NL_HOLIDAYS = holidays.Netherlands(years=range(2015, 2030))  # type: ignore[attr-defined]

logger = get_logger(__name__)


class GlobalScaler:
    """Z-score scaler using global statistics. Compatible with sklearn's StandardScaler."""
    
    def __init__(self, eps: float = 1e-8):
        self.mean_: np.ndarray = np.array([0.0])  # sklearn compatibility
        self.scale_: np.ndarray = np.array([1.0])  # sklearn compatibility (std)
        self.eps = eps
        self._fitted = False
    
    @property
    def mean(self) -> float:
        return float(self.mean_[0])
    
    @property
    def std(self) -> float:
        return float(self.scale_[0])
    
    def fit(self, values: np.ndarray) -> "GlobalScaler":
        """Fit scaler on all values."""
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
    
    def transform(self, value: float) -> float:
        """Transform a single value."""
        return (value - self.mean) / self.std
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform array (sklearn compatible)."""
        X = np.asarray(X)
        return X * self.scale_ + self.mean_
    
    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse transform a tensor."""
        return tensor * self.std + self.mean
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted


def compute_time_context_vectorized(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Vectorized time context: ``[n, 12]`` with sin/cos tod, dow one-hot, sin/cos month, holiday."""
    # Time of day: vectorized sin/cos encoding
    minutes = timestamps.hour * 60 + timestamps.minute
    tod_fraction = minutes / 1440.0
    sin_tod = np.sin(2 * np.pi * tod_fraction)
    cos_tod = np.cos(2 * np.pi * tod_fraction)
    
    # Day of week: vectorized one-hot encoding
    dow = timestamps.dayofweek.values  # 0=Monday, 6=Sunday
    dow_onehot = np.zeros((len(timestamps), 7), dtype=np.float32)
    dow_onehot[np.arange(len(timestamps)), dow] = 1.0
    
    # Month: vectorized sin/cos encoding (circular seasonal signal)
    month_fraction = (timestamps.month - 1) / 12.0
    sin_month = np.sin(2 * np.pi * month_fraction)
    cos_month = np.cos(2 * np.pi * month_fraction)
    
    # Holiday flag: check unique days only, then broadcast back (avoids per-row Python loop)
    day_keys = timestamps.normalize()
    unique_days = day_keys.unique()
    holiday_lookup = pd.Series(
        [1.0 if d.date() in _NL_HOLIDAYS else 0.0 for d in unique_days],
        index=unique_days,
        dtype=np.float32,
    )
    is_holiday = day_keys.map(holiday_lookup).to_numpy(dtype=np.float32)
    
    # Combine all features (ablation: exclude month/holiday via env vars)
    import os
    parts = [sin_tod, cos_tod, dow_onehot]
    if os.environ.get("ABLATE_NO_MONTH") != "1":
        parts.extend([sin_month, cos_month])
    if os.environ.get("ABLATE_NO_HOLIDAY") != "1":
        parts.append(is_holiday)
    time_context = np.column_stack(parts)
    return time_context.astype(np.float32)


class ProxyTaskBatchIterator:
    """Batch iterator that slices pre-computed dataset tensors directly.

    Yields ``(link_indices, link_ctx, link_targets, stop_indices, stop_ctx,
    stop_targets, stop_rate)``. ``subsample < 1.0`` draws a fresh random
    subset each iteration (requires ``shuffle=True``).
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
        self.subsample = subsample
        self._n_link = dataset.num_link_samples
        self._n_stop = dataset.num_stop_samples
        self._n_total = self._n_link + self._n_stop
        self._n_used = int(self._n_total * subsample)

    def __len__(self) -> int:
        return (self._n_used + self.batch_size - 1) // self.batch_size

    def __iter__(self):  # noqa: ANN204
        if self.shuffle:
            perm = torch.randperm(self._n_total)[: self._n_used]
        else:
            perm = torch.arange(self._n_total)

        ds = self.dataset
        ctx_dim = ds._link_time_context.shape[1]

        for start in range(0, self._n_used, self.batch_size):
            batch_idx = perm[start : start + self.batch_size]

            link_mask = batch_idx < self._n_link
            stop_mask = ~link_mask

            link_sel = batch_idx[link_mask]
            stop_sel = batch_idx[stop_mask] - self._n_link

            # links
            if link_sel.numel() > 0:
                link_indices = ds._link_node_indices[link_sel]
                link_ctx = ds._link_time_context[link_sel]
                link_targets = ds._link_targets[link_sel]
            else:
                link_indices = torch.empty(0, dtype=torch.long)
                link_ctx = torch.empty(0, ctx_dim, dtype=torch.float32)
                link_targets = torch.empty(0, dtype=torch.float32)

            # stops
            if stop_sel.numel() > 0:
                stop_indices = ds._stop_node_indices[stop_sel]
                stop_ctx = ds._stop_time_context[stop_sel]
                stop_targets = ds._stop_targets[stop_sel]
                stop_rate: Optional[torch.Tensor] = ds._stop_rate_targets[stop_sel]
            else:
                stop_indices = torch.empty(0, dtype=torch.long)
                stop_ctx = torch.empty(0, ctx_dim, dtype=torch.float32)
                stop_targets = torch.empty(0, dtype=torch.float32)
                stop_rate = None

            if self.pin_memory:
                link_indices = link_indices.pin_memory()
                link_ctx = link_ctx.pin_memory()
                link_targets = link_targets.pin_memory()
                stop_indices = stop_indices.pin_memory()
                stop_ctx = stop_ctx.pin_memory()
                stop_targets = stop_targets.pin_memory()
                if stop_rate is not None:
                    stop_rate = stop_rate.pin_memory()

            yield (
                link_indices, link_ctx, link_targets,
                stop_indices, stop_ctx, stop_targets,
                stop_rate,
            )


class RawObservationProxyDataset:
    """Dataset using raw individual observations (not 15-min aggregates).

    Each dwell observation yields a binary stopped/pass-through classification target
    and a regression target (dwell duration). Link observations yield travel time targets.
    Supports freeflow residual and ratio modes for link targets.
    """
    
    def __init__(
        self,
        df_travel: pd.DataFrame,
        df_dwell: pd.DataFrame,
        link_nodes: pd.DataFrame,
        stop_nodes: pd.DataFrame,
        link_scaler: Optional[GlobalScaler] = None,
        stop_scaler: Optional[GlobalScaler] = None,
        log_transform: bool = False,
        freeflow_times: Optional[dict[str, float]] = None,
        freeflow_ratio: bool = False,
    ):
        
        if freeflow_ratio and log_transform:
            raise ValueError(
                "freeflow_ratio=True and log_transform=True cannot be combined. "
                "The ratio mode produces multiplicative targets that are not "
                "compatible with the log-space encoding used by log_transform."
            )
        
        # Validate required columns
        required_travel_cols = {"link", "from_time", "travel_time"}
        missing_travel = required_travel_cols - set(df_travel.columns)
        if missing_travel:
            raise ValueError(f"df_travel missing required columns: {missing_travel}")
        
        required_dwell_cols = {"stop", "from_time", "travel_time"}
        missing_dwell = required_dwell_cols - set(df_dwell.columns)
        if missing_dwell:
            raise ValueError(f"df_dwell missing required columns: {missing_dwell}")
        
        self.log_transform = log_transform
        self.freeflow_times = freeflow_times
        self.freeflow_ratio = freeflow_ratio
        if self.log_transform:
            logger.debug("Log transform enabled")
        if self.freeflow_times is not None:
            if self.freeflow_ratio:
                logger.debug("Freeflow ratio mode")
            else:
                logger.debug("Freeflow residual mode")
        
        self.link_scaler = link_scaler if link_scaler is not None else GlobalScaler()
        self.stop_scaler = stop_scaler if stop_scaler is not None else GlobalScaler()

        link_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(link_nodes["id"])}
        stop_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(stop_nodes["id"])}

        # Link (travel time) samples
        logger.debug("Building link samples from %d travel time records", len(df_travel))
        
        df_travel = df_travel.copy()
        df_travel["link_str"] = df_travel["link"].astype(str)
        valid_travel_mask = df_travel["link_str"].isin(link_id_to_idx)
        df_travel_valid = df_travel[valid_travel_mask]

        if len(df_travel_valid) < len(df_travel):
            logger.debug("Filtered travel times: %d -> %d (removed links not in graph)",
                        len(df_travel), len(df_travel_valid))

        if len(df_travel_valid) == 0:
            raise ValueError("No travel time observations match graph link nodes. "
                           "Check that df_travel['link'] values match link_nodes['id'].")
        
        link_travel_times = df_travel_valid["travel_time"].to_numpy().copy()

        if self.freeflow_times is not None:
            ff_values = df_travel_valid["link_str"].map(self.freeflow_times).to_numpy(dtype=np.float64)
            missing_ff = np.isnan(ff_values)
            if missing_ff.any():
                n_missing = missing_ff.sum()
                missing_ids = df_travel_valid.loc[missing_ff, "link_str"].unique()  # type: ignore[union-attr]
                raise ValueError(
                    f"Missing freeflow times for {n_missing} observations "
                    f"({len(missing_ids)} unique links). "
                    f"First few: {list(missing_ids[:5])}"
                )
            if self.freeflow_ratio:
                ff_safe = np.maximum(ff_values, 1.0)  # avoid div-by-zero
                link_values = link_travel_times / ff_safe
                self._link_freeflow_refs = torch.from_numpy(ff_safe.astype(np.float32))
            elif self.log_transform:
                link_values = np.log1p(link_travel_times) - np.log1p(ff_values)
                self._link_freeflow_refs = torch.from_numpy(np.log1p(ff_values).astype(np.float32))
            else:
                link_values = link_travel_times - ff_values
                self._link_freeflow_refs = torch.from_numpy(ff_values.astype(np.float32))
        else:
            if self.log_transform:
                link_values = np.log1p(link_travel_times)
            else:
                link_values = link_travel_times.copy()
            self._link_freeflow_refs = None
        
        if not self.link_scaler.is_fitted:
            self.link_scaler.fit(link_values)
            logger.debug("Fitted link scaler: mean=%.2f, std=%.2f",
                        self.link_scaler.mean, self.link_scaler.std)

        link_node_indices = df_travel_valid["link_str"].map(link_id_to_idx).values
        self._link_node_indices = torch.from_numpy(link_node_indices.astype(np.int64))
        self._link_municipalities: Optional[np.ndarray] = (
            df_travel_valid["municipality"].values
            if "municipality" in df_travel_valid.columns else None
        )
        
        normalized_link_times = (link_values - self.link_scaler.mean) / self.link_scaler.std
        self._link_targets = torch.from_numpy(normalized_link_times.astype(np.float32))
        
        link_timestamps = pd.DatetimeIndex(df_travel_valid["from_time"])
        self._link_time_context = torch.from_numpy(
            compute_time_context_vectorized(link_timestamps)
        )
        
        self._num_link_samples = len(df_travel_valid)
        
        # Stop (dwell time) samples
        logger.debug("Building stop samples from %d dwell time records", len(df_dwell))
        df_dwell = df_dwell.copy()
        df_dwell["stop_str"] = "stop_" + df_dwell["stop"].astype(str)
        valid_dwell_mask = df_dwell["stop_str"].isin(stop_id_to_idx)
        df_dwell_valid = df_dwell[valid_dwell_mask]

        if len(df_dwell_valid) < len(df_dwell):
            logger.debug("Filtered dwell times: %d -> %d (removed stops not in graph)",
                        len(df_dwell), len(df_dwell_valid))
        
        if len(df_dwell_valid) == 0:
            raise ValueError("No dwell time observations match graph stop nodes. "
                           "Check that df_dwell['stop'] values match stop_nodes['id'].")
        
        dwell_times = df_dwell_valid["travel_time"].to_numpy()  # column is named travel_time
        stopped_mask = dwell_times > 0

        n_stopped = stopped_mask.sum()
        pct_stopped = n_stopped / len(stopped_mask) * 100
        logger.debug("Dwell observations: %d stopped (%.1f%%)", n_stopped, pct_stopped)
        
        if self.log_transform:
            dwell_times = np.log1p(dwell_times)

        # Fit on stopped observations only (excludes pass-through zeros)
        if not self.stop_scaler.is_fitted:
            stopped_dwell_values = dwell_times[stopped_mask]
            if len(stopped_dwell_values) == 0:
                raise ValueError("No stopped observations (dwell > 0) found for scaler fitting")
            self.stop_scaler.fit(stopped_dwell_values)
            logger.debug("Fitted stop scaler: mean=%.2f, std=%.2f",
                        self.stop_scaler.mean, self.stop_scaler.std)

        stop_node_indices = df_dwell_valid["stop_str"].map(stop_id_to_idx).values
        self._stop_node_indices = torch.from_numpy(stop_node_indices.astype(np.int64))
        
        normalized_dwell_times = (dwell_times - self.stop_scaler.mean) / self.stop_scaler.std
        self._stop_targets = torch.from_numpy(normalized_dwell_times.astype(np.float32))
        self._stop_rate_targets = torch.from_numpy(stopped_mask.astype(np.float32))
        # Municipalities aligned with stopped observations only (model predicts dwell
        # regression for stopped obs; classification targets use all stop obs).
        self._stop_municipalities: Optional[np.ndarray] = (
            df_dwell_valid["municipality"].values[stopped_mask]
            if "municipality" in df_dwell_valid.columns else None
        )
        
        stop_timestamps = pd.DatetimeIndex(df_dwell_valid["from_time"])
        self._stop_time_context = torch.from_numpy(
            compute_time_context_vectorized(stop_timestamps)
        )
        
        self._num_stop_samples = len(df_dwell_valid)
        
        logger.info("Created %d link samples, %d stop samples (from raw individual observations)",
                   self._num_link_samples, self._num_stop_samples)
    
    def __len__(self) -> int:
        return self._num_link_samples + self._num_stop_samples
    
    @property
    def has_stop_rate(self) -> bool:
        """Whether this dataset has stop_rate classification targets (always True)."""
        return True

    @property
    def link_municipalities(self) -> Optional[np.ndarray]:
        """Municipality label per link observation, aligned with model predictions."""
        return self._link_municipalities

    @property
    def stop_municipalities(self) -> Optional[np.ndarray]:
        """Municipality label per stopped dwell observation, aligned with model predictions."""
        return self._stop_municipalities
    
    @property
    def num_link_samples(self) -> int:
        return self._num_link_samples
    
    @property
    def num_stop_samples(self) -> int:
        return self._num_stop_samples
    
    @property
    def link_freeflow_refs(self) -> Optional[torch.Tensor]:
        """Per-sample freeflow references for reversing residuals to absolute times."""
        return self._link_freeflow_refs
