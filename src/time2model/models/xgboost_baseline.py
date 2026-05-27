"""XGBoost baseline that reuses the GNN train/eval flow.

Reads features from the same HeteroData graphs and runs on top of
RawObservationProxyDataset so the target transforms and filters match the
GNN exactly. The wrapper is an ``nn.Module`` so the shared ``evaluate`` path
works unchanged; predictions stay in scaled target space and the GNN's
inverse-transform (scaler + freeflow undo + clamp) is applied downstream.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from torch_geometric.data import HeteroData

from time2model.data.proxy_task_dataset import RawObservationProxyDataset
from time2model.logging import get_logger

logger = get_logger(__name__)


class XGBoostBaseline(nn.Module):
    """Flat XGBoost regressor + classifier with a GNN-compatible forward signature."""

    MODEL_TYPE = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self._common_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
        )
        self._early_stopping_rounds = early_stopping_rounds

        self.link_model: Optional[xgb.XGBRegressor] = None
        self.stop_reg_model: Optional[xgb.XGBRegressor] = None
        self.stop_cls_model: Optional[xgb.XGBClassifier] = None

        # nn.Module needs at least one tensor for .to(device) to be a no-op.
        self.register_buffer("_dummy", torch.zeros(1))

    def fit(
        self,
        train_dataset: RawObservationProxyDataset,
        train_graph: HeteroData,
        val_dataset: Optional[RawObservationProxyDataset] = None,
        val_graph: Optional[HeteroData] = None,
    ) -> None:
        """Fit link regressor, stop regressor (stopped-only), stop classifier.

        Stopped-only regression matches the GNN, which only collects
        ``stop_preds[stopped_mask]`` for its stop regression metrics.
        """
        self._fit_link(train_dataset, train_graph, val_dataset, val_graph)
        self._fit_stop(train_dataset, train_graph, val_dataset, val_graph)

    def _fit_link(
        self,
        train_dataset: RawObservationProxyDataset,
        train_graph: HeteroData,
        val_dataset: Optional[RawObservationProxyDataset],
        val_graph: Optional[HeteroData],
    ) -> None:
        X_train, y_train = _build_link_xy(train_dataset, train_graph)
        eval_set = None
        if (
            val_dataset is not None
            and val_graph is not None
            and val_dataset.num_link_samples > 0
        ):
            X_val, y_val = _build_link_xy(val_dataset, val_graph)
            eval_set = [(X_val, y_val)]

        params = dict(self._common_params)
        if eval_set is not None:
            params["early_stopping_rounds"] = self._early_stopping_rounds
        self.link_model = xgb.XGBRegressor(**params)
        logger.info(
            "Fitting link XGBoost on %d observations (%d features)",
            len(y_train), X_train.shape[1],
        )
        self.link_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        if eval_set is not None:
            logger.info("Link best iteration: %d", self.link_model.best_iteration)

    def _fit_stop(
        self,
        train_dataset: RawObservationProxyDataset,
        train_graph: HeteroData,
        val_dataset: Optional[RawObservationProxyDataset],
        val_graph: Optional[HeteroData],
    ) -> None:
        X_train, y_train, stopped_mask = _build_stop_xy(train_dataset, train_graph)
        y_stopped_bin = stopped_mask.astype(np.int32)

        # Validation tensors (shared between regression and classification fits)
        X_val: Optional[np.ndarray] = None
        y_val_dwell: Optional[np.ndarray] = None
        y_val_stopped_bin: Optional[np.ndarray] = None
        val_stopped_mask: Optional[np.ndarray] = None
        if (
            val_dataset is not None
            and val_graph is not None
            and val_dataset.num_stop_samples > 0
        ):
            X_val, y_val_dwell, val_stopped_mask = _build_stop_xy(val_dataset, val_graph)
            y_val_stopped_bin = val_stopped_mask.astype(np.int32)

        # Regression (stopped-only)
        X_train_stopped = X_train[stopped_mask]
        y_train_stopped = y_train[stopped_mask]
        reg_eval_set = None
        if X_val is not None and val_stopped_mask is not None and val_stopped_mask.any():
            reg_eval_set = [(X_val[val_stopped_mask], y_val_dwell[val_stopped_mask])]

        params = dict(self._common_params)
        if reg_eval_set is not None:
            params["early_stopping_rounds"] = self._early_stopping_rounds
        self.stop_reg_model = xgb.XGBRegressor(**params)
        logger.info(
            "Fitting stop regressor (stopped-only) on %d/%d observations",
            len(y_train_stopped), len(y_train),
        )
        self.stop_reg_model.fit(
            X_train_stopped, y_train_stopped, eval_set=reg_eval_set, verbose=False,
        )
        if reg_eval_set is not None:
            logger.info("Stop reg best iteration: %d", self.stop_reg_model.best_iteration)

        # Classification (stopped vs pass-through)
        cls_eval_set = None
        if X_val is not None and y_val_stopped_bin is not None:
            cls_eval_set = [(X_val, y_val_stopped_bin)]

        params = dict(self._common_params)
        params.update(objective="binary:logistic", eval_metric="logloss")
        if cls_eval_set is not None:
            params["early_stopping_rounds"] = self._early_stopping_rounds
        self.stop_cls_model = xgb.XGBClassifier(**params)
        logger.info(
            "Fitting stop classifier on %d observations (%.1f%% stopped)",
            len(y_stopped_bin), 100.0 * y_stopped_bin.mean(),
        )
        self.stop_cls_model.fit(
            X_train, y_stopped_bin, eval_set=cls_eval_set, verbose=False,
        )
        if cls_eval_set is not None:
            logger.info("Stop cls best iteration: %d", self.stop_cls_model.best_iteration)

    def forward(  # type: ignore[override]
        self,
        batch: HeteroData,
        link_node_indices: torch.Tensor,
        link_time_context: torch.Tensor,
        stop_node_indices: torch.Tensor,
        stop_time_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict in scaled target space (matches GNN's pre-inverse-transform output)."""
        device = self._dummy.device

        if link_node_indices.numel() > 0:
            assert self.link_model is not None, "Call fit() before forward()"
            X = _stack_features(batch["link"].x, link_node_indices, link_time_context)
            preds = self.link_model.predict(X)
            link_preds = torch.from_numpy(np.asarray(preds, dtype=np.float32)).to(device)
        else:
            link_preds = torch.empty(0, device=device)

        if stop_node_indices.numel() > 0:
            assert self.stop_reg_model is not None and self.stop_cls_model is not None
            X = _stack_features(batch["stop"].x, stop_node_indices, stop_time_context)
            reg_preds = self.stop_reg_model.predict(X)
            # Raw margins so evaluate()'s sigmoid → probability matches the GNN path.
            margin = self.stop_cls_model.predict(X, output_margin=True)
            stop_preds = torch.from_numpy(np.asarray(reg_preds, dtype=np.float32)).to(device)
            stop_logits = torch.from_numpy(np.asarray(margin, dtype=np.float32)).to(device)
        else:
            stop_preds = torch.empty(0, device=device)
            stop_logits = torch.empty(0, device=device)

        # No embeddings — oversmoothing is GNN-only and skipped upstream.
        return link_preds, stop_preds, stop_logits, {}


def _stack_features(
    node_x: torch.Tensor,
    indices: torch.Tensor,
    time_context: torch.Tensor,
) -> np.ndarray:
    """Concatenate static node features with per-observation time context.

    ``evaluate`` puts HeteroData on the model device, but dataset index
    tensors stay on CPU. Pulling ``node_x`` back to CPU before indexing both
    fixes the device mismatch and skips an unnecessary GPU↔CPU round-trip
    (XGBoost runs on CPU anyway).
    """
    static = node_x.cpu()[indices.cpu()].numpy()
    ctx = time_context.cpu().numpy()
    return np.hstack([static, ctx]).astype(np.float32)


def _build_link_xy(
    dataset: RawObservationProxyDataset,
    graph: HeteroData,
) -> Tuple[np.ndarray, np.ndarray]:
    X = _stack_features(
        graph["link"].x,
        dataset._link_node_indices,
        dataset._link_time_context,
    )
    y = dataset._link_targets.cpu().numpy().astype(np.float32)
    return X, y


def _build_stop_xy(
    dataset: RawObservationProxyDataset,
    graph: HeteroData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _stack_features(
        graph["stop"].x,
        dataset._stop_node_indices,
        dataset._stop_time_context,
    )
    y = dataset._stop_targets.cpu().numpy().astype(np.float32)
    stopped_mask = dataset._stop_rate_targets.cpu().numpy().astype(bool)
    return X, y, stopped_mask
