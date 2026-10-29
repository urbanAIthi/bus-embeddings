"""Evaluation metrics: embedding diversity (MAD, MADGap) and seasonal MASE / RMSSE."""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from bus_embeddings.logging import get_logger

_logger = get_logger(__name__)

# Seasonal period of the naive baseline (one week).
_SEASONAL_PERIOD = pd.Timedelta(days=7)


# =========================================================================
# MAD / MADGap (Chen et al., 2020)
# =========================================================================


def compute_mad(embeddings: torch.Tensor) -> float:
    """Mean average cosine distance over all pairs of node embeddings (0–2)."""
    if embeddings.size(0) < 2:
        return 0.0

    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    distance_matrix = 1 - torch.mm(embeddings_norm, embeddings_norm.t())

    # Unique pairs: upper triangle without the diagonal
    num_nodes = embeddings.size(0)
    mask = torch.triu(
        torch.ones(num_nodes, num_nodes, device=embeddings.device),
        diagonal=1
    ).bool()
    return distance_matrix[mask].mean().item()


def compute_mad_gap(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[float, float, float]:
    """Return ``(mad_gap, mad_connected, mad_remote)``; MADGap = MAD_remote - MAD_connected."""
    num_nodes = embeddings.size(0)
    if num_nodes < 2 or edge_index.size(1) == 0:
        return 0.0, 0.0, 0.0

    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    distance_matrix = 1 - torch.mm(embeddings_norm, embeddings_norm.t())

    # Symmetric adjacency without self-loops
    adj_mask = torch.zeros(num_nodes, num_nodes, device=embeddings.device, dtype=torch.bool)
    src, dst = edge_index[0], edge_index[1]
    adj_mask[src, dst] = True
    adj_mask[dst, src] = True
    diag_mask = torch.eye(num_nodes, device=embeddings.device, dtype=torch.bool)
    adj_mask = adj_mask & ~diag_mask
    non_adj_mask = ~adj_mask & ~diag_mask

    mad_connected = distance_matrix[adj_mask].mean().item() if adj_mask.sum() > 0 else 0.0
    mad_remote = distance_matrix[non_adj_mask].mean().item() if non_adj_mask.sum() > 0 else 0.0
    return mad_remote - mad_connected, mad_connected, mad_remote


def compute_oversmoothing_metrics(
    model: nn.Module,
    graph_data: HeteroData,
    device: torch.device,
) -> Dict[str, float]:
    """MAD and MADGap of the link and stop embeddings of ``graph_data``.

    Two links are neighbours if they share a stop; two stops are neighbours
    if they share a link.
    """
    model.eval()
    graph_data = graph_data.to(device)
    with torch.no_grad():
        embeddings = model.compute_embeddings(graph_data)

    link_emb = embeddings["link"]
    stop_emb = embeddings["stop"]
    num_links = link_emb.size(0)
    num_stops = stop_emb.size(0)

    metrics: Dict[str, float] = {
        "link/mad": compute_mad(link_emb),
        "stop/mad": compute_mad(stop_emb),
    }

    # Link-stop incidence matrix [num_links, num_stops]
    link_to_src_edge = graph_data["link", "to_src", "stop"].edge_index
    link_to_dst_edge = graph_data["link", "to_dst", "stop"].edge_index
    link_stop_adj = torch.zeros(num_links, num_stops, device=device, dtype=torch.float32)
    link_stop_adj[
        link_to_src_edge[0].clamp(0, num_links - 1),
        link_to_src_edge[1].clamp(0, num_stops - 1),
    ] = 1.0
    link_stop_adj[
        link_to_dst_edge[0].clamp(0, num_links - 1),
        link_to_dst_edge[1].clamp(0, num_stops - 1),
    ] = 1.0

    for node_type, emb, adj in (
        ("link", link_emb, torch.mm(link_stop_adj, link_stop_adj.t()) > 0),
        ("stop", stop_emb, torch.mm(link_stop_adj.t(), link_stop_adj) > 0),
    ):
        adj.fill_diagonal_(False)
        edge_index = adj.nonzero(as_tuple=False).t()
        if edge_index.size(1) > 0:
            mad_gap, _, _ = compute_mad_gap(emb, edge_index)
        else:
            mad_gap = 0.0
        metrics[f"{node_type}/mad_gap"] = mad_gap

    return metrics


# =========================================================================
# Seasonal MASE / RMSSE
# =========================================================================


def _seasonal_naive_errors(
    df_eval: pd.DataFrame,
    df_history: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(actuals, forecasts)`` of the weekly seasonal naive forecast.

    The forecast for an observation is the most recent observation of the
    same node at or before one week earlier.
    """
    eval_df = df_eval[[id_col, time_col, value_col]].copy()
    eval_df["_ts_lag"] = eval_df[time_col] - _SEASONAL_PERIOD

    hist = df_history[[id_col, time_col, value_col]].copy()
    hist = hist.rename(columns={value_col: "_naive_value"})

    merged = pd.merge_asof(
        eval_df.sort_values("_ts_lag"),
        hist.sort_values(time_col),
        left_on="_ts_lag",
        right_on=time_col,
        by=id_col,
        direction="backward",
        suffixes=("", "_hist"),
    )

    matched = merged["_naive_value"].notna()
    if not matched.any():
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    actuals = merged.loc[matched, value_col].values.astype(np.float64)
    forecasts = merged.loc[matched, "_naive_value"].values.astype(np.float64)
    return actuals, forecasts


def compute_seasonal_naive_metrics(
    df_eval: pd.DataFrame,
    df_history: pd.DataFrame,
    id_col: str,
    model_mae: float,
    model_rmse: float,
    time_col: str = "from_time",
    value_col: str = "travel_time",
) -> Dict[str, float]:
    """Seasonal MASE and RMSSE of a model relative to the weekly seasonal naive forecast.

    ``mase = MAE_model / MAE_naive`` and ``rmsse = RMSE_model / RMSE_naive``,
    where the naive errors are computed on the observations of ``df_eval``
    using ``df_history`` (which must cover the week before ``df_eval``) as
    the source of the forecasts.

    Returns:
        Dict with ``naive_mae``, ``naive_rmse``, ``naive_n_matched``, ``mase``
        and ``rmsse``; empty if no observation has a week-old counterpart.
    """
    actuals, naive_forecasts = _seasonal_naive_errors(
        df_eval=df_eval,
        df_history=df_history,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
    )
    if len(actuals) == 0:
        _logger.warning("No week-ago matches for the seasonal naive forecast (%s).", id_col)
        return {}

    naive_errors = actuals - naive_forecasts
    naive_mae = float(np.mean(np.abs(naive_errors)))
    naive_rmse = float(np.sqrt(np.mean(naive_errors ** 2)))

    result: Dict[str, float] = {
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_n_matched": len(actuals),
    }
    if naive_mae > 0:
        result["mase"] = model_mae / naive_mae
    if naive_rmse > 0:
        result["rmsse"] = model_rmse / naive_rmse
    return result


def compute_seasonal_naive_metrics_by_region(
    df_eval: pd.DataFrame,
    df_history: pd.DataFrame,
    id_col: str,
    model_preds: np.ndarray,
    model_targets: np.ndarray,
    model_regions: np.ndarray,
    time_col: str = "from_time",
    value_col: str = "travel_time",
    region_col: str = "municipality",
) -> Dict[str, Any]:
    """Seasonal MASE / RMSSE per region, combined with the geometric mean.

    Args:
        df_eval: Evaluation observations with a ``region_col`` column.
        df_history: Observations covering the week before ``df_eval``.
        id_col: Node identifier column (``"link"`` or ``"stop"``).
        model_preds: Model predictions in seconds, aligned with ``model_regions``.
        model_targets: Observed values in seconds, aligned with ``model_regions``.
        model_regions: Region of each prediction.

    Returns:
        Dict with ``mase`` and ``rmsse`` (geometric means over regions) and
        ``naive_n_matched``.
    """
    mase_vals: list[float] = []
    rmsse_vals: list[float] = []
    n_matched = 0

    for region in np.unique(model_regions):
        mask = model_regions == region
        preds_r = model_preds[mask]
        targets_r = model_targets[mask]
        model_mae_r = float(np.mean(np.abs(preds_r - targets_r)))
        model_rmse_r = float(np.sqrt(np.mean((preds_r - targets_r) ** 2)))

        sn = compute_seasonal_naive_metrics(
            df_eval=df_eval[df_eval[region_col] == region],
            df_history=df_history[df_history[region_col] == region],
            id_col=id_col,
            model_mae=model_mae_r,
            model_rmse=model_rmse_r,
            time_col=time_col,
            value_col=value_col,
        )
        if not sn:
            continue
        n_matched += int(sn["naive_n_matched"])
        if "mase" in sn:
            mase_vals.append(sn["mase"])
        if "rmsse" in sn:
            rmsse_vals.append(sn["rmsse"])
        _logger.debug(
            "%s %s: MASE=%.3f RMSSE=%.3f",
            id_col, region, sn.get("mase", float("nan")), sn.get("rmsse", float("nan")),
        )

    result: Dict[str, Any] = {"naive_n_matched": n_matched}
    if mase_vals:
        result["mase"] = float(np.exp(np.mean(np.log(mase_vals))))
    if rmsse_vals:
        result["rmsse"] = float(np.exp(np.mean(np.log(rmsse_vals))))
    return result
