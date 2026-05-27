"""Heterogeneous bipartite graph construction and time series utilities."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from time2model.features import get_enabled_columns
from time2model.logging import get_logger

logger = get_logger(__name__)

# Columns to EXCLUDE from the feature tensor.
# Only truly zero-variance features that carry no information.
EXCLUDE_FEATURES: frozenset[str] = frozenset({
    # Link: lane fractions with zero or near-zero variance
    "osm_lanes_7_frac",          # 100% zero (1 nonzero value in 4516 links)
    "osm_lanes_8_frac",          # 100% zero (all zeros)
    # Stop: POI categories with no data at all
    "osm_poi_sports_recreation_500m",  # 100% zero
    "osm_poi_tourism_500m",            # 100% zero
})


# Columns to log1p-transform BEFORE z-score normalization.
# Features where log makes semantic sense (multiplicative scale or counts)
# and skew is high enough to distort z-score normalization.
#
# Link features:
#   - osm_length:       distance (m), skew 6.80
#   - osm_travel_time:  freeflow time (s), skew 4.61
#   - gtfs_frequency_1d: daily trips, skew 4.57
# Stop features:
#   - osm_poi_*_500m:   POI counts, all right-skewed
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




@dataclass
class FeatureNormStats:
    """Z-score normalization statistics for graph node features, persisted in checkpoints."""

    link_mean: Optional[pd.Series] = None
    link_std: Optional[pd.Series] = None
    stop_mean: Optional[pd.Series] = None
    stop_std: Optional[pd.Series] = None
    link_columns: List[str] = field(default_factory=list)
    stop_columns: List[str] = field(default_factory=list)


def normalize(
    df: pd.DataFrame, cols: list,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score normalize; constant columns are zeroed out (std stored as 0)."""
    mu = df[cols].mean()
    sd = df[cols].std()
    # For columns with nonzero std, apply z-score; for constant columns, output 0.
    safe_sd = sd.replace(0.0, 1.0)  # avoid division by zero; result masked below
    out = (df[cols] - mu) / safe_sd
    # Explicitly zero-out constant columns (std == 0)
    constant_mask = sd == 0.0
    if constant_mask.any():
        constant_cols = constant_mask[constant_mask].index.tolist()
        out[constant_cols] = 0.0
    return out, mu, sd


def apply_normalize(
    df: pd.DataFrame,
    cols: list,
    mu: pd.Series,
    sd: pd.Series,
) -> pd.DataFrame:
    """Apply pre-computed z-score statistics; constant columns (sd=0) are zeroed."""
    missing = set(cols) - set(mu.index)
    if missing:
        raise ValueError(
            f"Columns {missing} are not present in the pre-computed "
            f"normalization statistics.  Available: {list(mu.index)}"
        )
    safe_sd = sd[cols].replace(0.0, 1.0)  # avoid division by zero
    out = (df[cols] - mu[cols]) / safe_sd
    # Zero-out columns that were constant in the training set
    constant_mask = sd[cols] == 0.0
    if constant_mask.any():
        constant_cols = constant_mask[constant_mask].index.tolist()
        out[constant_cols] = 0.0
    return out


def _derive_route_shared_edges(
    stop_src_of_link_ei: torch.Tensor,
    stop_dst_of_link_ei: torch.Tensor,
    num_links: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive route-shared link↔link and stop↔stop edges from the bipartite stop↔link edges.

    - link↔link: two links share an edge iff they touch a common stop (as src or dst).
      The cartesian merge below yields both directions naturally (i,j) and (j,i).
    - stop↔stop: two stops share an edge iff some link connects them as (src, dst);
      both directions are added explicitly.

    Self-loops are dropped.  Duplicate edges (e.g., two links share both endpoints)
    are collapsed to a single edge per direction.
    """
    src_np = stop_src_of_link_ei.cpu().numpy()  # row 0 = stop_idx, row 1 = link_idx
    dst_np = stop_dst_of_link_ei.cpu().numpy()

    # stop↔stop: for each link with both endpoints known, emit (src_stop, dst_stop)
    link_to_src_stop = np.full(num_links, -1, dtype=np.int64)
    link_to_dst_stop = np.full(num_links, -1, dtype=np.int64)
    if src_np.size:
        link_to_src_stop[src_np[1]] = src_np[0]
    if dst_np.size:
        link_to_dst_stop[dst_np[1]] = dst_np[0]
    valid = (link_to_src_stop >= 0) & (link_to_dst_stop >= 0)
    ss_a = link_to_src_stop[valid]
    ss_b = link_to_dst_stop[valid]
    not_self = ss_a != ss_b
    ss_a, ss_b = ss_a[not_self], ss_b[not_self]
    if ss_a.size:
        ss_pairs = pd.DataFrame({"s": np.concatenate([ss_a, ss_b]),
                                 "d": np.concatenate([ss_b, ss_a])}).drop_duplicates()
        stop_stop_ei = torch.tensor(
            np.stack([ss_pairs["s"].to_numpy(), ss_pairs["d"].to_numpy()]),
            dtype=torch.long,
        )
    else:
        stop_stop_ei = torch.zeros((2, 0), dtype=torch.long)

    # link↔link: group all (stop, link) incidences by stop and self-join to produce
    # every pair of links touching the same stop.
    stop_arr = np.concatenate([src_np[0], dst_np[0]]) if (src_np.size or dst_np.size) else np.empty(0, dtype=np.int64)
    link_arr = np.concatenate([src_np[1], dst_np[1]]) if (src_np.size or dst_np.size) else np.empty(0, dtype=np.int64)
    if stop_arr.size:
        incidences = pd.DataFrame({"stop": stop_arr, "link": link_arr}).drop_duplicates()
        joined = incidences.merge(incidences, on="stop", suffixes=("_i", "_j"))
        joined = joined[joined["link_i"] != joined["link_j"]]
        pairs = joined[["link_i", "link_j"]].drop_duplicates()
        link_link_ei = torch.tensor(
            np.stack([pairs["link_i"].to_numpy(), pairs["link_j"].to_numpy()]),
            dtype=torch.long,
        ) if len(pairs) else torch.zeros((2, 0), dtype=torch.long)
    else:
        link_link_ei = torch.zeros((2, 0), dtype=torch.long)

    return link_link_ei, stop_stop_ei


def build_hetero_graph(
    link_nodes_csv: Union[str, Path, List[Union[str, Path]]],
    stop_nodes_csv: Union[str, Path, List[Union[str, Path]]],
    edges_csv: Union[str, Path, List[Union[str, Path]]],
    feature_config: Dict[str, bool],
    feature_norm_stats: Optional[FeatureNormStats] = None,
) -> Tuple[HeteroData, pd.DataFrame, pd.DataFrame, FeatureNormStats]:
    """Build HeteroData from bipartite graph CSVs (supports multi-region)."""
    # Handle single paths or lists of paths
    link_paths = [link_nodes_csv] if isinstance(link_nodes_csv, (str, Path)) else link_nodes_csv
    stop_paths = [stop_nodes_csv] if isinstance(stop_nodes_csv, (str, Path)) else stop_nodes_csv
    edge_paths = [edges_csv] if isinstance(edges_csv, (str, Path)) else edges_csv
    
    # Validate inputs
    if not link_paths or not stop_paths or not edge_paths:
        raise ValueError("At least one path must be provided for each of link_nodes, stop_nodes, and edges")
    
    if not (len(link_paths) == len(stop_paths) == len(edge_paths)):
        raise ValueError(
            f"Number of paths must match for all inputs. "
            f"Got {len(link_paths)} link paths, {len(stop_paths)} stop paths, {len(edge_paths)} edge paths"
        )
    
    num_regions = len(link_paths)
    logger.info("Building hetero graph from %d region(s)", num_regions)
    
    # Load and concatenate node and edge data from all regions
    all_link_nodes = []
    all_stop_nodes = []
    all_edges = []
    
    link_offset = 0
    stop_offset = 0
    
    for region_idx, (link_path, stop_path, edge_path) in enumerate(zip(link_paths, stop_paths, edge_paths)):
        # Load CSVs for this region
        link_nodes = pd.read_csv(link_path)
        stop_nodes = pd.read_csv(stop_path)
        edges = pd.read_csv(edge_path)

        logger.debug("Region %d: %d link nodes, %d stop nodes, %d edges",
                    region_idx, len(link_nodes), len(stop_nodes), len(edges))
        
        # Add region offset to maintain unique global indices
        link_nodes = link_nodes.copy()
        stop_nodes = stop_nodes.copy()
        edges = edges.copy()
        
        # Store original index as 'local_index' before offsetting
        link_nodes["local_index"] = range(len(link_nodes))
        stop_nodes["local_index"] = range(len(stop_nodes))
        
        # Add region identifier
        link_nodes["region"] = region_idx
        stop_nodes["region"] = region_idx
        
        # Build local ID to local index mappings for this region
        link_to_local_idx = dict(zip(link_nodes["id"].astype(str), link_nodes["local_index"]))
        stop_to_local_idx = dict(zip(stop_nodes["id"].astype(str), stop_nodes["local_index"]))
        
        # Process edges: map IDs to local indices, then add global offset
        edges["src"] = edges["src"].astype(str)
        edges["dst"] = edges["dst"].astype(str)
        
        # Map stop_src_of_link edges (source stop → link)
        stop_src_of_link = edges[edges["type"] == "stop_src_of_link"].copy()
        stop_src_of_link["src_idx"] = stop_src_of_link["src"].map(stop_to_local_idx)
        stop_src_of_link["dst_idx"] = stop_src_of_link["dst"].map(link_to_local_idx)
        n_before = len(stop_src_of_link)
        stop_src_of_link = stop_src_of_link.dropna(subset=["src_idx", "dst_idx"])
        n_dropped = n_before - len(stop_src_of_link)
        if n_dropped > 0:
            logger.warning("Region %d: dropped %d/%d stop_src_of_link edges with unmatched IDs",
                          region_idx, n_dropped, n_before)
        stop_src_of_link["src_idx"] = stop_src_of_link["src_idx"].astype(int) + stop_offset
        stop_src_of_link["dst_idx"] = stop_src_of_link["dst_idx"].astype(int) + link_offset
        
        # Map stop_dst_of_link edges (destination stop → link)
        stop_dst_of_link = edges[edges["type"] == "stop_dst_of_link"].copy()
        stop_dst_of_link["src_idx"] = stop_dst_of_link["src"].map(stop_to_local_idx)
        stop_dst_of_link["dst_idx"] = stop_dst_of_link["dst"].map(link_to_local_idx)
        n_before = len(stop_dst_of_link)
        stop_dst_of_link = stop_dst_of_link.dropna(subset=["src_idx", "dst_idx"])
        n_dropped = n_before - len(stop_dst_of_link)
        if n_dropped > 0:
            logger.warning("Region %d: dropped %d/%d stop_dst_of_link edges with unmatched IDs",
                          region_idx, n_dropped, n_before)
        stop_dst_of_link["src_idx"] = stop_dst_of_link["src_idx"].astype(int) + stop_offset
        stop_dst_of_link["dst_idx"] = stop_dst_of_link["dst_idx"].astype(int) + link_offset
        
        # Map link_to_src_stop edges (link → source stop, reverse)
        link_to_src = edges[edges["type"] == "link_to_src_stop"].copy()
        link_to_src["src_idx"] = link_to_src["src"].map(link_to_local_idx)
        link_to_src["dst_idx"] = link_to_src["dst"].map(stop_to_local_idx)
        n_before = len(link_to_src)
        link_to_src = link_to_src.dropna(subset=["src_idx", "dst_idx"])
        n_dropped = n_before - len(link_to_src)
        if n_dropped > 0:
            logger.warning("Region %d: dropped %d/%d link_to_src_stop edges with unmatched IDs",
                          region_idx, n_dropped, n_before)
        link_to_src["src_idx"] = link_to_src["src_idx"].astype(int) + link_offset
        link_to_src["dst_idx"] = link_to_src["dst_idx"].astype(int) + stop_offset
        
        # Map link_to_dst_stop edges (link → destination stop, reverse)
        link_to_dst = edges[edges["type"] == "link_to_dst_stop"].copy()
        link_to_dst["src_idx"] = link_to_dst["src"].map(link_to_local_idx)
        link_to_dst["dst_idx"] = link_to_dst["dst"].map(stop_to_local_idx)
        n_before = len(link_to_dst)
        link_to_dst = link_to_dst.dropna(subset=["src_idx", "dst_idx"])
        n_dropped = n_before - len(link_to_dst)
        if n_dropped > 0:
            logger.warning("Region %d: dropped %d/%d link_to_dst_stop edges with unmatched IDs",
                          region_idx, n_dropped, n_before)
        link_to_dst["src_idx"] = link_to_dst["src_idx"].astype(int) + link_offset
        link_to_dst["dst_idx"] = link_to_dst["dst_idx"].astype(int) + stop_offset
        
        # Combine edges for this region
        region_edges = pd.concat(
            [stop_src_of_link, stop_dst_of_link, link_to_src, link_to_dst],
            ignore_index=True,
        )
        
        all_link_nodes.append(link_nodes)
        all_stop_nodes.append(stop_nodes)
        all_edges.append(region_edges)
        
        # Update offsets for next region
        link_offset += len(link_nodes)
        stop_offset += len(stop_nodes)
    
    # Concatenate all regions
    link_nodes_combined = pd.concat(all_link_nodes, ignore_index=True)
    stop_nodes_combined = pd.concat(all_stop_nodes, ignore_index=True)
    edges_combined = pd.concat(all_edges, ignore_index=True)
    
    # Create global index column (used for PyG node ordering)
    link_nodes_combined["index"] = range(len(link_nodes_combined))
    stop_nodes_combined["index"] = range(len(stop_nodes_combined))
    
    logger.info("Combined graph: %d link nodes, %d stop nodes, %d edges",
                len(link_nodes_combined), len(stop_nodes_combined), len(edges_combined))
    
    # Process link node features
    link_metadata_cols = {"id", "node_type", "src", "dst", "index", "local_index", "region",
                          "from_geometry", "to_geometry"}
    link_available_cols = [c for c in link_nodes_combined.columns if c not in link_metadata_cols]
    link_enabled_cols = get_enabled_columns(feature_config, link_available_cols)
    # Remove zero/near-zero variance features that hurt spatial transfer
    _excluded_link = [c for c in link_enabled_cols if c in EXCLUDE_FEATURES]
    if _excluded_link:
        link_enabled_cols = [c for c in link_enabled_cols if c not in EXCLUDE_FEATURES]
        logger.debug("Excluded %d near-zero-variance link features: %s", len(_excluded_link), _excluded_link)
    
    if link_enabled_cols:
        logger.debug("Link node features: %s", ", ".join(link_enabled_cols[:8]) + 
                   ("..." if len(link_enabled_cols) > 8 else ""))
        link_feature_df = link_nodes_combined[link_enabled_cols].copy()
        for c in link_enabled_cols:
            link_feature_df[c] = pd.to_numeric(link_feature_df[c], errors="coerce").fillna(0.0)
        # Apply log1p to extremely skewed features BEFORE z-score normalization
        log_cols = [c for c in link_enabled_cols if c in LOG_TRANSFORM_FEATURES]
        if log_cols:
            for c in log_cols:
                link_feature_df[c] = np.log1p(link_feature_df[c].clip(lower=0))
            logger.debug("Applied log1p pre-transform to link features: %s", log_cols)
        if feature_norm_stats is not None and feature_norm_stats.link_mean is not None:
            logger.debug("Applying pre-computed link feature normalization stats")
            link_feature_norm = apply_normalize(
                link_feature_df, link_enabled_cols,
                feature_norm_stats.link_mean, feature_norm_stats.link_std,
            )
            link_mu, link_sd = feature_norm_stats.link_mean, feature_norm_stats.link_std
        else:
            link_feature_norm, link_mu, link_sd = normalize(link_feature_df, link_enabled_cols)
        link_x = torch.tensor(link_feature_norm.values, dtype=torch.float)
    else:
        logger.debug("No static features enabled for link nodes")
        link_x = torch.zeros(len(link_nodes_combined), 0, dtype=torch.float)
        link_mu, link_sd = None, None
    
    # Process stop node features
    stop_metadata_cols = {"id", "node_type", "stop_id", "index", "local_index", "region",
                          "geometry"}
    stop_available_cols = [c for c in stop_nodes_combined.columns if c not in stop_metadata_cols]
    stop_enabled_cols = get_enabled_columns(feature_config, stop_available_cols)
    # Remove zero/near-zero variance features that hurt spatial transfer
    _excluded_stop = [c for c in stop_enabled_cols if c in EXCLUDE_FEATURES]
    if _excluded_stop:
        stop_enabled_cols = [c for c in stop_enabled_cols if c not in EXCLUDE_FEATURES]
        logger.debug("Excluded %d near-zero-variance stop features: %s", len(_excluded_stop), _excluded_stop)
    
    # For stops, we use lon/lat if use-location is enabled
    if feature_config.get("use-location", False):
        stop_loc_cols = ["lon", "lat"]
        stop_enabled_cols = [c for c in stop_loc_cols if c in stop_nodes_combined.columns] + \
                           [c for c in stop_enabled_cols if c not in stop_loc_cols]
    
    if stop_enabled_cols:
        logger.debug("Stop node features: %s", ", ".join(stop_enabled_cols[:8]) + 
                   ("..." if len(stop_enabled_cols) > 8 else ""))
        stop_feature_df = stop_nodes_combined[stop_enabled_cols].copy()
        for c in stop_enabled_cols:
            stop_feature_df[c] = pd.to_numeric(stop_feature_df[c], errors="coerce").fillna(0.0)
        # Apply log1p to extremely skewed features BEFORE z-score normalization
        log_cols = [c for c in stop_enabled_cols if c in LOG_TRANSFORM_FEATURES]
        if log_cols:
            for c in log_cols:
                stop_feature_df[c] = np.log1p(stop_feature_df[c].clip(lower=0))
            logger.debug("Applied log1p pre-transform to stop features: %s", log_cols)
        if feature_norm_stats is not None and feature_norm_stats.stop_mean is not None:
            logger.debug("Applying pre-computed stop feature normalization stats")
            stop_feature_norm = apply_normalize(
                stop_feature_df, stop_enabled_cols,
                feature_norm_stats.stop_mean, feature_norm_stats.stop_std,
            )
            stop_mu, stop_sd = feature_norm_stats.stop_mean, feature_norm_stats.stop_std
        else:
            stop_feature_norm, stop_mu, stop_sd = normalize(stop_feature_df, stop_enabled_cols)
        stop_x = torch.tensor(stop_feature_norm.values, dtype=torch.float)
    else:
        logger.debug("No static features enabled for stop nodes")
        stop_x = torch.zeros(len(stop_nodes_combined), 0, dtype=torch.float)
        stop_mu, stop_sd = None, None
    
    # Extract edges — 4 directed edge types
    def _extract_edge_index(df: pd.DataFrame, edge_type: str) -> torch.Tensor:
        subset = df[df["type"] == edge_type]
        return torch.tensor(
            np.vstack([
                subset["src_idx"].astype(int).to_numpy(),
                subset["dst_idx"].astype(int).to_numpy(),
            ]),
            dtype=torch.long,
        )
    
    stop_src_of_link_ei = _extract_edge_index(edges_combined, "stop_src_of_link")
    stop_dst_of_link_ei = _extract_edge_index(edges_combined, "stop_dst_of_link")
    link_to_src_stop_ei = _extract_edge_index(edges_combined, "link_to_src_stop")
    link_to_dst_stop_ei = _extract_edge_index(edges_combined, "link_to_dst_stop")
    
    logger.debug(
        "Edge indices: stop_src_of_link=%d, stop_dst_of_link=%d, "
        "link_to_src_stop=%d, link_to_dst_stop=%d",
        stop_src_of_link_ei.shape[1], stop_dst_of_link_ei.shape[1],
        link_to_src_stop_ei.shape[1], link_to_dst_stop_ei.shape[1],
    )
    
    # Create HeteroData object
    data = HeteroData()
    
    # Node features
    data["link"].x = link_x
    data["stop"].x = stop_x
    
    # Edge indices — 4 directed types preserving endpoint roles
    data["stop", "src_of", "link"].edge_index = stop_src_of_link_ei
    data["stop", "dst_of", "link"].edge_index = stop_dst_of_link_ei
    data["link", "to_src", "stop"].edge_index = link_to_src_stop_ei
    data["link", "to_dst", "stop"].edge_index = link_to_dst_stop_ei

    # Route-shared homogeneous edges, derived purely from the bipartite routing.
    # Adds (link, near, link) and (stop, near, stop) so the GNN can aggregate
    # neighborhood signal in one hop instead of two through the bipartite graph.
    if feature_config.get("add-route-shared-edges", False):
        link_link_ei, stop_stop_ei = _derive_route_shared_edges(
            stop_src_of_link_ei, stop_dst_of_link_ei, num_links=len(link_nodes_combined),
        )
        data["link", "near", "link"].edge_index = link_link_ei
        data["stop", "near", "stop"].edge_index = stop_stop_ei
        logger.info(
            "Added route-shared edges: link-near-link=%d, stop-near-stop=%d",
            link_link_ei.shape[1], stop_stop_ei.shape[1],
        )

    logger.info("HeteroData: link features=%s, stop features=%s",
                link_x.shape, stop_x.shape)
    
    # Build normalization stats (either freshly computed or passed through)
    norm_stats = FeatureNormStats(
        link_mean=link_mu,
        link_std=link_sd,
        stop_mean=stop_mu,
        stop_std=stop_sd,
        link_columns=list(link_enabled_cols),
        stop_columns=list(stop_enabled_cols),
    )
    
    return data, link_nodes_combined, stop_nodes_combined, norm_stats

