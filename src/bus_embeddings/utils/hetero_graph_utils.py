"""Assembly of the heterogeneous link/stop graph from the graph artifacts."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from bus_embeddings.features import get_enabled_columns
from bus_embeddings.logging import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

# Features without information (all zero) that are excluded from the inputs.
EXCLUDE_FEATURES: frozenset[str] = frozenset({
    "osm_lanes_7_frac",
    "osm_lanes_8_frac",
    "osm_poi_sports_recreation_500m",
    "osm_poi_tourism_500m",
})

# Skewed distance, duration and count features that are log1p-transformed
# before the z-score normalization.
LOG_TRANSFORM_FEATURES: frozenset[str] = frozenset({
    "osm_length",
    "osm_travel_time",
    "gtfs_frequency_1d",
    "osm_poi_food_drink_500m",
    "osm_poi_shopping_500m",
    "osm_poi_education_500m",
    "osm_poi_healthcare_500m",
    "osm_poi_transport_500m",
    "osm_poi_finance_500m",
    "osm_poi_entertainment_500m",
    "osm_poi_services_500m",
    "osm_poi_worship_500m",
    "osm_poi_other_amenity_500m",
})

# (edge type in edges.csv, source node type, relation, target node type)
EDGE_TYPES: Tuple[Tuple[str, str, str, str], ...] = (
    ("stop_src_of_link", "stop", "src_of", "link"),
    ("stop_dst_of_link", "stop", "dst_of", "link"),
    ("link_to_src_stop", "link", "to_src", "stop"),
    ("link_to_dst_stop", "link", "to_dst", "stop"),
)


@dataclass
class FeatureNormStats:
    """Z-score statistics of the node features, fitted on the training regions."""

    link_mean: Optional[pd.Series] = None
    link_std: Optional[pd.Series] = None
    stop_mean: Optional[pd.Series] = None
    stop_std: Optional[pd.Series] = None
    link_columns: List[str] = field(default_factory=list)
    stop_columns: List[str] = field(default_factory=list)


def normalize(
    df: pd.DataFrame, cols: list,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score normalize; constant columns are set to zero (std stored as 0)."""
    mu = df[cols].mean()
    sd = df[cols].std()
    out = (df[cols] - mu) / sd.replace(0.0, 1.0)
    constant_mask = sd == 0.0
    if constant_mask.any():
        out[constant_mask[constant_mask].index.tolist()] = 0.0
    return out, mu, sd


def apply_normalize(
    df: pd.DataFrame,
    cols: list,
    mu: pd.Series,
    sd: pd.Series,
) -> pd.DataFrame:
    """Apply pre-computed z-score statistics; columns constant in training are set to zero."""
    missing = set(cols) - set(mu.index)
    if missing:
        raise ValueError(
            f"Columns {missing} are not present in the normalization statistics. "
            f"Available: {list(mu.index)}"
        )
    out = (df[cols] - mu[cols]) / sd[cols].replace(0.0, 1.0)
    constant_mask = sd[cols] == 0.0
    if constant_mask.any():
        out[constant_mask[constant_mask].index.tolist()] = 0.0
    return out


def _feature_matrix(
    nodes: pd.DataFrame,
    feature_config: Dict[str, bool],
    mean: Optional[pd.Series],
    std: Optional[pd.Series],
) -> Tuple[torch.Tensor, List[str], Optional[pd.Series], Optional[pd.Series]]:
    """Return the normalized feature matrix, its columns and the z-score statistics."""
    cols = [
        c for c in get_enabled_columns(feature_config, list(nodes.columns))
        if c not in EXCLUDE_FEATURES
    ]
    if not cols:
        return torch.zeros(len(nodes), 0, dtype=torch.float), cols, None, None

    df = nodes[cols].copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in cols:
        if c in LOG_TRANSFORM_FEATURES:
            df[c] = np.log1p(df[c].clip(lower=0))

    if mean is not None and std is not None:
        normalized = apply_normalize(df, cols, mean, std)
    else:
        normalized, mean, std = normalize(df, cols)
    return torch.tensor(normalized.values, dtype=torch.float), cols, mean, std


def build_hetero_graph(
    link_nodes_csv: Union[PathLike, Sequence[PathLike]],
    stop_nodes_csv: Union[PathLike, Sequence[PathLike]],
    edges_csv: Union[PathLike, Sequence[PathLike]],
    feature_config: Dict[str, bool],
    feature_norm_stats: Optional[FeatureNormStats] = None,
) -> Tuple[HeteroData, pd.DataFrame, pd.DataFrame, FeatureNormStats]:
    """Build one graph from the artifacts of one or more municipalities.

    Node features are z-scored with ``feature_norm_stats`` if given, otherwise
    with statistics computed on this graph (returned for reuse).

    Returns:
        The graph, its link and stop nodes (row order = node index), and the
        feature normalization statistics.
    """
    link_paths = [link_nodes_csv] if isinstance(link_nodes_csv, (str, Path)) else list(link_nodes_csv)
    stop_paths = [stop_nodes_csv] if isinstance(stop_nodes_csv, (str, Path)) else list(stop_nodes_csv)
    edge_paths = [edges_csv] if isinstance(edges_csv, (str, Path)) else list(edges_csv)
    if not link_paths or not (len(link_paths) == len(stop_paths) == len(edge_paths)):
        raise ValueError(
            "Expected the same non-zero number of link node, stop node and edge files, got "
            f"{len(link_paths)}, {len(stop_paths)} and {len(edge_paths)}"
        )

    all_link_nodes: List[pd.DataFrame] = []
    all_stop_nodes: List[pd.DataFrame] = []
    all_edges: List[pd.DataFrame] = []
    offsets = {"link": 0, "stop": 0}

    for region_idx, (link_path, stop_path, edge_path) in enumerate(zip(link_paths, stop_paths, edge_paths)):
        link_nodes = pd.read_csv(link_path)
        stop_nodes = pd.read_csv(stop_path)
        edges = pd.read_csv(edge_path)
        edges["src"] = edges["src"].astype(str)
        edges["dst"] = edges["dst"].astype(str)

        local_index = {
            "link": dict(zip(link_nodes["id"].astype(str), range(len(link_nodes)))),
            "stop": dict(zip(stop_nodes["id"].astype(str), range(len(stop_nodes)))),
        }

        # Map node IDs to indices in the combined graph
        for csv_type, src_type, _, dst_type in EDGE_TYPES:
            typed = edges[edges["type"] == csv_type].copy()
            typed["src_idx"] = typed["src"].map(local_index[src_type])
            typed["dst_idx"] = typed["dst"].map(local_index[dst_type])
            n_before = len(typed)
            typed = typed.dropna(subset=["src_idx", "dst_idx"])
            if len(typed) < n_before:
                logger.warning("Region %d: dropped %d/%d %s edges with unmatched IDs",
                               region_idx, n_before - len(typed), n_before, csv_type)
            typed["src_idx"] = typed["src_idx"].astype(int) + offsets[src_type]
            typed["dst_idx"] = typed["dst_idx"].astype(int) + offsets[dst_type]
            all_edges.append(typed)

        all_link_nodes.append(link_nodes)
        all_stop_nodes.append(stop_nodes)
        offsets["link"] += len(link_nodes)
        offsets["stop"] += len(stop_nodes)

    link_nodes_combined = pd.concat(all_link_nodes, ignore_index=True)
    stop_nodes_combined = pd.concat(all_stop_nodes, ignore_index=True)
    edges_combined = pd.concat(all_edges, ignore_index=True)

    stats = feature_norm_stats or FeatureNormStats()
    link_x, link_cols, link_mu, link_sd = _feature_matrix(
        link_nodes_combined, feature_config, stats.link_mean, stats.link_std,
    )
    stop_x, stop_cols, stop_mu, stop_sd = _feature_matrix(
        stop_nodes_combined, feature_config, stats.stop_mean, stats.stop_std,
    )

    data = HeteroData()
    data["link"].x = link_x
    data["stop"].x = stop_x
    for csv_type, src_type, relation, dst_type in EDGE_TYPES:
        typed = edges_combined[edges_combined["type"] == csv_type]
        data[src_type, relation, dst_type].edge_index = torch.tensor(
            np.vstack([
                typed["src_idx"].astype(int).to_numpy(),
                typed["dst_idx"].astype(int).to_numpy(),
            ]),
            dtype=torch.long,
        )

    logger.info(
        "Graph of %d region(s): %d link nodes (%d features), %d stop nodes (%d features), %d edges",
        len(link_paths), link_x.shape[0], link_x.shape[1],
        stop_x.shape[0], stop_x.shape[1], len(edges_combined),
    )

    norm_stats = FeatureNormStats(
        link_mean=link_mu,
        link_std=link_sd,
        stop_mean=stop_mu,
        stop_std=stop_sd,
        link_columns=list(link_cols),
        stop_columns=list(stop_cols),
    )
    return data, link_nodes_combined, stop_nodes_combined, norm_stats
