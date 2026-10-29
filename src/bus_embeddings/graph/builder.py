"""Builder for the bipartite transit graph of link and stop nodes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from shapely import wkt

from bus_embeddings.graph.census_features import GISCOFeatureExtractor, LanduseFeatureExtractor
from bus_embeddings.graph.gtfs_features import GTFSFeatureExtractor
from bus_embeddings.graph.osm_features import OSMFeatureExtractor
from bus_embeddings.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GraphArtifacts:
    """Node and edge tables of a bipartite graph."""
    link_nodes: pd.DataFrame
    stop_nodes: pd.DataFrame
    edges: pd.DataFrame


def _geometric_median(points: np.ndarray, eps: float = 1e-9, max_iter: int = 200) -> np.ndarray:
    """Weiszfeld's algorithm. ``points`` is (N, 2). Returns the (x, y) median."""
    if len(points) == 1:
        return points[0].copy()
    y = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - y, axis=1)
        mask = d > eps
        if not mask.any():
            return y
        w = 1.0 / d[mask]
        new_y = (points[mask] * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(new_y - y) < eps:
            return new_y
        y = new_y
    return y


def _build_stop_coord_map(travel_df: pd.DataFrame) -> Dict[str, tuple]:
    """Aggregate per-stop coordinates from travel observations via geometric median.

    Each stop typically appears in many rows. Observations are pooled across
    ``from_stop``/``to_stop``, parsed once per unique WKT, and the geometric
    median (weighted by occurrence count) is taken, which is robust to
    occasional outlier positions for the same stop_id. Unparseable WKT raises
    ``ValueError``.
    """
    parts = [
        travel_df[['from_stop', 'from_geometry']].rename(
            columns={'from_stop': 'stop', 'from_geometry': 'geometry'}
        ),
        travel_df[['to_stop', 'to_geometry']].rename(
            columns={'to_stop': 'stop', 'to_geometry': 'geometry'}
        ),
    ]
    all_pts = pd.concat(parts, ignore_index=True).dropna(subset=['geometry'])

    parsed: Dict[str, tuple] = {}
    for g in all_pts['geometry'].unique():
        try:
            pt = wkt.loads(g)
            parsed[g] = (pt.x, pt.y)
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Failed to parse geometry {g!r}: {e}") from e

    all_pts['lon'] = all_pts['geometry'].map(lambda g: parsed[g][0])
    all_pts['lat'] = all_pts['geometry'].map(lambda g: parsed[g][1])

    coord_map: Dict[str, tuple] = {}
    for stop_id, group in all_pts.groupby('stop'):
        pts = group[['lon', 'lat']].to_numpy()
        median = _geometric_median(pts)
        coord_map[stop_id] = (float(median[0]), float(median[1]))
    return coord_map


def _validate_nodes(nodes: pd.DataFrame) -> None:
    """Validate node DataFrame for duplicates and NaN values."""
    if nodes["id"].duplicated().any():
        raise ValueError("Duplicate node_id detected in nodes.")
    for col in nodes.columns:
        if nodes[col].isna().any():
            raise ValueError(f"NaN values detected in node column: {col}")


def _concat_features(nodes: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Append feature columns to row-aligned node rows."""
    return pd.concat([nodes.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


class BipartiteGraphBuilder:
    """Build a bipartite graph with separate link and stop nodes.

    Link nodes get Valhalla road features and GTFS link features; stop nodes
    get OSMnx POI and landuse features, GISCO population density and GTFS stop
    features.
    """

    def __init__(
        self,
        gtfs_path: Path,
        census_path: Path,
        valhalla_url: str = "http://localhost:8002",
        osm_poi_buffer_m: float = 500.0,
        osm_projected_crs: str = "EPSG:28992",
    ):
        self._osm_extractor = OSMFeatureExtractor(
            poi_buffer_m=osm_poi_buffer_m,
            projected_crs=osm_projected_crs,
            valhalla_url=valhalla_url,
        )
        # Landuse extractor shares buffer/CRS with OSM extractor
        self._landuse_extractor = LanduseFeatureExtractor(
            buffer_m=osm_poi_buffer_m,
            projected_crs=osm_projected_crs,
        )
        self._gtfs_extractor = GTFSFeatureExtractor(gtfs_path=gtfs_path)
        self._gisco_extractor = GISCOFeatureExtractor(census_path=census_path)

    def build(self, travel_times: pd.DataFrame) -> GraphArtifacts:
        """Build bipartite graph from travel time observations."""
        coord_map = _build_stop_coord_map(travel_times)
        link_nodes = self._build_link_nodes(travel_times, coord_map)
        stop_nodes = self._build_stop_nodes(coord_map)

        logger.info("Enriching link nodes with road features from Valhalla...")
        road_features = self._osm_extractor.extract_road_features(link_nodes)
        link_nodes = _concat_features(link_nodes, road_features)
        logger.info("Added %d road feature columns to link nodes", len(road_features.columns))

        logger.info("Enriching stop nodes with POI features from OSMnx...")
        poi_features = self._osm_extractor.extract_poi_features_for_stops(stop_nodes)
        stop_nodes = _concat_features(stop_nodes, poi_features)
        logger.info("Added %d POI feature columns to stop nodes", len(poi_features.columns))

        logger.info("Enriching stop nodes with landuse features from OSMnx...")
        landuse_features = self._landuse_extractor.extract_features(stop_nodes)
        stop_nodes = _concat_features(stop_nodes, landuse_features)
        logger.info("Added %d landuse feature columns to stop nodes", len(landuse_features.columns))

        logger.info("Enriching stop nodes with GISCO census features...")
        gisco_features = self._gisco_extractor.extract_features(stop_nodes)
        stop_nodes = _concat_features(stop_nodes, gisco_features)
        logger.info("Added %d GISCO feature columns to stop nodes", len(gisco_features.columns))

        logger.info("Enriching stop nodes with GTFS features...")
        gtfs_features = self._gtfs_extractor.extract_features(stop_nodes)
        stop_nodes = _concat_features(stop_nodes, gtfs_features)
        logger.info("Added %d GTFS feature columns to stop nodes", len(gtfs_features.columns))

        logger.info("Enriching link nodes with GTFS features...")
        gtfs_link_features = self._gtfs_extractor.extract_link_features(link_nodes)
        link_nodes = _concat_features(link_nodes, gtfs_link_features)
        logger.info("Added %d GTFS feature columns to link nodes", len(gtfs_link_features.columns))

        _validate_nodes(link_nodes)
        _validate_nodes(stop_nodes)

        edges = self._build_bipartite_edges(link_nodes)

        return GraphArtifacts(link_nodes=link_nodes, stop_nodes=stop_nodes, edges=edges)

    def _build_link_nodes(self, travel_df: pd.DataFrame, coord_map: Dict[str, tuple]) -> pd.DataFrame:
        """Build link nodes (from_stop → to_stop) with geographic coordinates.

        Self-loops (from_stop == to_stop) are dropped: they represent dwell-time
        records leaking into the travel-time stream and have no valid OSM routing.
        """
        # Get unique links, excluding self-loops
        link_stats = travel_df[['from_stop', 'to_stop']].drop_duplicates().reset_index(drop=True)
        n_before = len(link_stats)
        link_stats = link_stats[link_stats['from_stop'] != link_stats['to_stop']].reset_index(drop=True)
        n_dropped = n_before - len(link_stats)
        if n_dropped > 0:
            logger.info("Dropped %d self-loop link nodes (from_stop == to_stop)", n_dropped)

        # Create link ID from from_stop and to_stop
        link_stats['id'] = link_stats.apply(
            lambda row: f"link_{row['from_stop']}_{row['to_stop']}", axis=1
        )

        missing = set(link_stats['from_stop']) | set(link_stats['to_stop'])
        missing -= coord_map.keys()
        if missing:
            raise ValueError(f"No geometry observed for stops: {sorted(missing)[:5]}")

        link_stats['from_lon'] = link_stats['from_stop'].map(lambda s: coord_map[s][0])
        link_stats['from_lat'] = link_stats['from_stop'].map(lambda s: coord_map[s][1])
        link_stats['to_lon'] = link_stats['to_stop'].map(lambda s: coord_map[s][0])
        link_stats['to_lat'] = link_stats['to_stop'].map(lambda s: coord_map[s][1])

        link_stats['src'] = link_stats['from_stop'].astype(str)
        link_stats['dst'] = link_stats['to_stop'].astype(str)
        link_stats['node_type'] = 'link'

        return link_stats[['id', 'node_type', 'src', 'dst', 'from_lon', 'from_lat', 'to_lon', 'to_lat']]

    def _build_stop_nodes(self, coord_map: Dict[str, tuple]) -> pd.DataFrame:
        """Build stop nodes from the per-stop coordinates."""
        all_stops = pd.DataFrame(
            [
                {'stop_id': stop_id, 'lon': lon, 'lat': lat}
                for stop_id, (lon, lat) in coord_map.items()
            ]
        )
        all_stops['id'] = all_stops['stop_id'].apply(lambda x: f"stop_{x}")
        all_stops['node_type'] = 'stop'
        return all_stops[['id', 'node_type', 'stop_id', 'lon', 'lat']]

    def _build_bipartite_edges(self, link_nodes: pd.DataFrame) -> pd.DataFrame:
        """Build directed endpoint-aware edges (4 types: stop_src_of_link, stop_dst_of_link, link_to_src_stop, link_to_dst_stop)."""
        edges = []

        for _, link in link_nodes.iterrows():
            src_stop = link['src']
            dst_stop = link['dst']
            link_id = link['id']

            # Source stop → link (this stop is the origin of this link)
            edges.append({
                'type': 'stop_src_of_link',
                'src': f"stop_{src_stop}",
                'dst': link_id,
            })

            # Destination stop → link (this stop is the destination of this link)
            edges.append({
                'type': 'stop_dst_of_link',
                'src': f"stop_{dst_stop}",
                'dst': link_id,
            })

            # Link → source stop (reverse of src_of)
            edges.append({
                'type': 'link_to_src_stop',
                'src': link_id,
                'dst': f"stop_{src_stop}",
            })

            # Link → destination stop (reverse of dst_of)
            edges.append({
                'type': 'link_to_dst_stop',
                'src': link_id,
                'dst': f"stop_{dst_stop}",
            })

        edges_df = pd.DataFrame(edges, columns=['type', 'src', 'dst'])

        logger.info(
            "Built %d bipartite edges: %d stop_src_of_link, %d stop_dst_of_link, "
            "%d link_to_src_stop, %d link_to_dst_stop",
            len(edges_df),
            len(edges_df[edges_df['type'] == 'stop_src_of_link']),
            len(edges_df[edges_df['type'] == 'stop_dst_of_link']),
            len(edges_df[edges_df['type'] == 'link_to_src_stop']),
            len(edges_df[edges_df['type'] == 'link_to_dst_stop']),
        )

        return edges_df
