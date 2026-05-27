"""Metrics for GNN embedding quality (oversmoothing) and seasonal MASE/RMSSE."""
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from time2model.logging import get_logger

_logger = get_logger(__name__)

# Seasonal period for the naive baseline (one week).
_SEASONAL_PERIOD = pd.Timedelta(days=7)


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for models that can compute node embeddings."""
    
    def compute_embeddings(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute embeddings for all node types."""
        ...
    
    def eval(self) -> Any:
        """Set model to evaluation mode."""
        ...


def compute_mad(embeddings: torch.Tensor) -> float:
    """Compute Mean Average Distance (cosine) for node embeddings. Returns 0–2."""
    if embeddings.size(0) < 2:
        return 0.0
    
    # Normalize embeddings for cosine distance computation
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise cosine similarity matrix: [num_nodes, num_nodes]
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    # Convert to cosine distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Get upper triangle (excluding diagonal) for unique pairs
    num_nodes = embeddings.size(0)
    mask = torch.triu(
        torch.ones(num_nodes, num_nodes, device=embeddings.device), 
        diagonal=1
    ).bool()
    
    # Compute mean of all pairwise distances
    mad = distance_matrix[mask].mean().item()
    
    return mad


def compute_mad_gap(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[float, float, float]:
    """Compute MADGap (MAD_remote - MAD_connected) to measure oversmoothing."""
    num_nodes = embeddings.size(0)
    if num_nodes < 2 or edge_index.size(1) == 0:
        return 0.0, 0.0, 0.0
    
    # Normalize embeddings for cosine distance
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute full pairwise cosine distance matrix
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    distance_matrix = 1 - similarity_matrix
    
    # Build adjacency mask from edge index (symmetric for undirected interpretation)
    adj_mask = torch.zeros(num_nodes, num_nodes, device=embeddings.device, dtype=torch.bool)
    src, dst = edge_index[0], edge_index[1]
    adj_mask[src, dst] = True
    adj_mask[dst, src] = True  # Make symmetric
    
    # Exclude self-loops from consideration
    diag_mask = torch.eye(num_nodes, device=embeddings.device, dtype=torch.bool)
    adj_mask = adj_mask & ~diag_mask
    
    # Non-connected pairs mask (excluding diagonal)
    non_adj_mask = ~adj_mask & ~diag_mask
    
    # Compute MAD for connected and remote pairs
    if adj_mask.sum() > 0:
        mad_connected = distance_matrix[adj_mask].mean().item()
    else:
        mad_connected = 0.0
    
    if non_adj_mask.sum() > 0:
        mad_remote = distance_matrix[non_adj_mask].mean().item()
    else:
        mad_remote = 0.0
    
    mad_gap = mad_remote - mad_connected
    
    return mad_gap, mad_connected, mad_remote


def compute_oversmoothing_metrics(
    model: EmbeddingModel,
    graph_data: HeteroData,
    device: torch.device,
) -> Dict[str, float]:
    """Compute MAD and MADGap oversmoothing metrics for both link and stop node types."""
    model.eval()
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        embeddings = model.compute_embeddings(graph_data)
    
    metrics: Dict[str, float] = {}
    
    # Extract embeddings
    link_emb = embeddings["link"]
    stop_emb = embeddings["stop"]
    num_links = link_emb.size(0)
    num_stops = stop_emb.size(0)
    
    # Compute global MAD for each node type
    metrics["link/mad"] = compute_mad(link_emb)
    metrics["stop/mad"] = compute_mad(stop_emb)
    
    # Build bipartite adjacency for same-type connectivity inference
    # In bipartite graphs: links connect via stops, stops connect via links
    # Use link→stop edges (both src and dst) to build adjacency
    link_to_src_edge = graph_data["link", "to_src", "stop"].edge_index
    link_to_dst_edge = graph_data["link", "to_dst", "stop"].edge_index
    
    # Build link-stop adjacency matrix [num_links, num_stops]
    link_stop_adj = torch.zeros(num_links, num_stops, device=device, dtype=torch.float32)
    link_stop_adj[
        link_to_src_edge[0].clamp(0, num_links - 1),
        link_to_src_edge[1].clamp(0, num_stops - 1),
    ] = 1.0
    link_stop_adj[
        link_to_dst_edge[0].clamp(0, num_links - 1),
        link_to_dst_edge[1].clamp(0, num_stops - 1),
    ] = 1.0
    
    # Link-link connectivity via shared stops: A @ A^T > 0
    link_link_adj = torch.mm(link_stop_adj, link_stop_adj.t()) > 0
    link_link_adj.fill_diagonal_(False)  # Remove self-connections
    link_link_edge = link_link_adj.nonzero(as_tuple=False).t()
    
    # Compute link MADGap
    if link_link_edge.size(1) > 0:
        mad_gap, mad_conn, mad_remote = compute_mad_gap(link_emb, link_link_edge)
    else:
        mad_gap, mad_conn, mad_remote = 0.0, 0.0, compute_mad(link_emb)
    
    metrics["link/mad_gap"] = mad_gap
    metrics["link/mad_connected"] = mad_conn
    metrics["link/mad_remote"] = mad_remote
    
    # Stop-stop connectivity via shared links: A^T @ A > 0
    stop_stop_adj = torch.mm(link_stop_adj.t(), link_stop_adj) > 0
    stop_stop_adj.fill_diagonal_(False)
    stop_stop_edge = stop_stop_adj.nonzero(as_tuple=False).t()
    
    # Compute stop MADGap
    if stop_stop_edge.size(1) > 0:
        mad_gap, mad_conn, mad_remote = compute_mad_gap(stop_emb, stop_stop_edge)
    else:
        mad_gap, mad_conn, mad_remote = 0.0, 0.0, compute_mad(stop_emb)
    
    metrics["stop/mad_gap"] = mad_gap
    metrics["stop/mad_connected"] = mad_conn
    metrics["stop/mad_remote"] = mad_remote
    
    return metrics


def compute_smape(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Symmetric MAPE as a percentage (0–200). Zero-zero pairs are excluded."""
    numerator = (y_pred - y_true).abs()
    denominator = (y_true.abs() + y_pred.abs()) / 2.0

    # Exclude degenerate pairs where both are zero
    valid = denominator > 0
    if not valid.any():
        _logger.warning("sMAPE: all denominators are zero; returning 0.0.")
        return 0.0

    smape = 100.0 * (numerator[valid] / denominator[valid]).mean().item()
    return smape


def compute_node_level_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    node_indices: torch.Tensor,
    time_contexts: "torch.Tensor | None" = None,
) -> Dict[str, float]:
    """Compute node×time-slot R², MAPE, and MAE.

    Observations are grouped by ``(node, hour, day_of_week)`` when
    *time_contexts* is provided, otherwise by ``node`` alone.
    This tests: "How long will link X take on Mondays at 8 AM?" —
    the real-world query a transit planner would make for an unseen region.

    Args:
        y_pred: Predicted values in original scale.  Shape ``(n,)``.
        y_true: Ground-truth values in original scale.  Shape ``(n,)``.
        node_indices: Node index for each observation.  Shape ``(n,)``.
        time_contexts: Optional time context vectors.  Shape ``(n, 12)``.
            Format: ``[sin_tod, cos_tod, dow_onehot(7), sin_month, cos_month, holiday]``.

    Returns:
        Dict with keys ``node_r2``, ``node_mape``, ``node_mae``,
        ``node_smape``, ``node_count``.
    """
    if y_pred.numel() == 0:
        return {"node_r2": 0.0, "node_mape": 0.0, "node_mae": 0.0,
                "node_smape": 0.0, "node_count": 0}

    pred_np = y_pred.detach().cpu().numpy().astype(float)
    true_np = y_true.detach().cpu().numpy().astype(float)
    idx_np = node_indices.detach().cpu().numpy()

    if time_contexts is not None and time_contexts.numel() > 0:
        ctx_np = time_contexts.detach().cpu().numpy()
        # Recover hour from sin/cos encoding: hour = atan2(sin, cos) / (2π) * 24
        hour_frac = np.arctan2(ctx_np[:, 0], ctx_np[:, 1]) / (2 * np.pi)
        hour_frac[hour_frac < 0] += 1.0  # Map to [0, 1)
        hours = (hour_frac * 24).astype(int) % 24

        # Day of week from one-hot (indices 2-8): 0=Mon … 6=Sun
        dow = np.argmax(ctx_np[:, 2:9], axis=1)

        # Composite key: node × hour × dow  (node*1000 + hour*10 + dow)
        # hour 0-23, dow 0-6 → hour*10+dow ranges 0-236, well within int64
        group_keys = idx_np.astype(np.int64) * 1000 + hours * 10 + dow
    else:
        group_keys = idx_np.astype(np.int64)

    unique_keys = np.unique(group_keys)
    n_groups = len(unique_keys)
    if n_groups < 2:
        return {"node_r2": 0.0, "node_mape": 0.0, "node_mae": 0.0,
                "node_smape": 0.0, "node_count": n_groups}

    # Per-group mean prediction and target (vectorized with pandas for speed)
    import pandas as pd
    df = pd.DataFrame({"key": group_keys, "pred": pred_np, "true": true_np})
    agg = df.groupby("key", sort=False).mean()
    group_pred = agg["pred"].values
    group_true = agg["true"].values

    # R²
    ss_res = ((group_pred - group_true) ** 2).sum()
    ss_tot = ((group_true - group_true.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE
    nonzero = np.abs(group_true) > 1e-8
    if nonzero.any():
        mape = float(100.0 * (np.abs(group_pred[nonzero] - group_true[nonzero])
                               / np.abs(group_true[nonzero])).mean())
    else:
        mape = 0.0

    # MAE
    mae = float(np.abs(group_pred - group_true).mean())

    # sMAPE
    denom = (np.abs(group_true) + np.abs(group_pred)) / 2.0
    valid = denom > 0
    if valid.any():
        smape = float(100.0 * (np.abs(group_pred[valid] - group_true[valid])
                                / denom[valid]).mean())
    else:
        smape = 0.0

    return {
        "node_r2": r2,
        "node_mape": mape,
        "node_mae": mae,
        "node_smape": smape,
        "node_count": n_groups,
    }


def _seasonal_naive_errors(
    df_eval: pd.DataFrame,
    df_history: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-observation seasonal naive errors using last-observation-carried-forward from one week prior."""
    # Prepare eval data with lag timestamp
    eval_df = df_eval[[id_col, time_col, value_col]].copy()
    eval_df["_ts_lag"] = eval_df[time_col] - _SEASONAL_PERIOD
    
    # Prepare history data (rename value column to avoid collision)
    hist = df_history[[id_col, time_col, value_col]].copy()
    hist = hist.rename(columns={value_col: "_naive_value"})
    
    # Use merge_asof to find the most recent historical value at or before lag time
    # This is much faster than nested loops - O(n log n) vs O(n^2)
    merged = pd.merge_asof(
        eval_df.sort_values("_ts_lag"),
        hist.sort_values(time_col),
        left_on="_ts_lag",
        right_on=time_col,
        by=id_col,
        direction="backward",
        suffixes=("", "_hist"),
    )
    
    # Filter to observations where a naive forecast was found
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
    time_col: str = "from_time",
    value_col: str = "travel_time",
    model_mae: float | None = None,
    model_rmse: float | None = None,
) -> Dict[str, float]:
    """Compute seasonal MASE / RMSSE against a one-week-prior naive baseline.

    Returns ``naive_mae`` / ``naive_rmse`` always; ``smase`` / ``srmsse`` only
    when ``model_mae`` / ``model_rmse`` are given.
    """
    actuals, naive_forecasts = _seasonal_naive_errors(
        df_eval=df_eval,
        df_history=df_history,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
    )

    result: Dict[str, float] = {}

    if len(actuals) == 0:
        _logger.warning(
            "No week-ago matches found for seasonal naive (%s); "
            "sMASE / sRMSSE cannot be computed.",
            id_col,
        )
        return result

    naive_errors = actuals - naive_forecasts
    naive_mae = float(np.mean(np.abs(naive_errors)))
    naive_rmse = float(np.sqrt(np.mean(naive_errors ** 2)))

    result["naive_mae"] = naive_mae
    result["naive_rmse"] = naive_rmse
    result["naive_n_matched"] = len(actuals)

    if naive_mae > 0 and model_mae is not None:
        result["smase"] = model_mae / naive_mae

    if naive_rmse > 0 and model_rmse is not None:
        result["srmsse"] = model_rmse / naive_rmse

    _logger.debug(
        "Seasonal naive (%s): MAE=%.4f, RMSE=%.4f (matched %d / %d eval obs)",
        id_col,
        naive_mae,
        naive_rmse,
        len(actuals),
        len(df_eval),
    )

    return result


def compute_smase_by_region(
    df_eval: pd.DataFrame,
    df_history: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    region_col: str,
    model_preds_orig: np.ndarray,
    model_targets_orig: np.ndarray,
    model_municipalities: np.ndarray,
) -> Dict[str, Any]:
    """Compute sMASE/sRMSSE as geometric mean across regions.

    Computes the ratio model_error / naive_error independently per region, then
    takes the geometric mean.  This prevents a single heavy-tailed region from
    dominating the pooled denominator (which is the failure mode of the simple
    model_RMSE / naive_RMSE ratio when regions have very different variance).

    Args:
        df_eval: Eval-period observations with *region_col* column.
        df_history: Full history (train + eval) with *region_col* column.
        id_col: Node identifier column (``"link"`` or ``"stop"``).
        time_col: Timestamp column name.
        value_col: Target value column name.
        region_col: Column that identifies the region (e.g. ``"municipality"``).
        model_preds_orig: Per-observation model predictions in original scale,
            aligned with *model_municipalities*.
        model_targets_orig: Per-observation targets in original scale.
        model_municipalities: Region label per prediction, same length as
            *model_preds_orig*.

    Returns:
        Dict with ``smase`` and ``srmsse`` (geometric mean across regions),
        ``naive_n_matched``, and ``per_region`` (raw per-region stats for
        diagnostics).
    """
    regions = np.unique(model_municipalities)
    smase_vals: list[float] = []
    srmsse_vals: list[float] = []
    per_region: Dict[str, Dict[str, float]] = {}

    for region in regions:
        mask = model_municipalities == region
        if not mask.any():
            continue

        preds_r = model_preds_orig[mask]
        targets_r = model_targets_orig[mask]
        model_mae_r = float(np.mean(np.abs(preds_r - targets_r)))
        model_rmse_r = float(np.sqrt(np.mean((preds_r - targets_r) ** 2)))

        df_eval_r = df_eval[df_eval[region_col] == region]
        df_hist_r = df_history[df_history[region_col] == region]

        sn = compute_seasonal_naive_metrics(
            df_eval=df_eval_r,
            df_history=df_hist_r,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            model_mae=model_mae_r,
            model_rmse=model_rmse_r,
        )

        if not sn:
            continue

        per_region[region] = {"model_mae": model_mae_r, "model_rmse": model_rmse_r, **sn}

        if "smase" in sn:
            smase_vals.append(sn["smase"])
        if "srmsse" in sn:
            srmsse_vals.append(sn["srmsse"])

    result: Dict[str, Any] = {"per_region": per_region}
    if smase_vals:
        result["smase"] = float(np.exp(np.mean(np.log(smase_vals))))
    if srmsse_vals:
        result["srmsse"] = float(np.exp(np.mean(np.log(srmsse_vals))))
    result["naive_n_matched"] = sum(
        int(v.get("naive_n_matched", 0)) for v in per_region.values()
    )

    _logger.info(
        "Per-region %s: %s → geomean sMASE=%.4f sRMSSE=%.4f",
        id_col,
        {r: f"sMASE={v.get('smase', float('nan')):.3f} sRMSSE={v.get('srmsse', float('nan')):.3f}"
         for r, v in per_region.items()},
        result.get("smase", float("nan")),
        result.get("srmsse", float("nan")),
    )

    return result
