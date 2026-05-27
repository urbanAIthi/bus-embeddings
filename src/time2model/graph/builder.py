"""Graph builders for link-only and bipartite graph models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from shapely import wkt

from .base import GraphArtifacts, GraphBuilder
from .census_features import (
    GISCOFeatureExtractor,
    LanduseFeatureExtractor,
    get_gisco_feature_columns,
    get_landuse_feature_columns,
)
from .gtfs_features import GTFSFeatureExtractor, get_gtfs_feature_columns, get_gtfs_link_feature_columns
from .osm_features import (
    OSMFeatureExtractor,
    get_osm_feature_columns,
    get_poi_feature_columns,
    get_road_feature_columns,
)
from ..logging import get_logger

logger = get_logger(__name__)


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
    median (weighted by occurrence count) is taken. The median is robust to
    occasional bad geometry strings for the same stop_id.
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


class LinkGraphBuilder(GraphBuilder):
    """Build a link graph where road segments are nodes connected via shared stops."""
    
    def __init__(
        self,
        enrich_osm: bool = False,
        osm_poi_buffer_m: float = 100.0,
        osm_projected_crs: str = "EPSG:28992",
        valhalla_url: str = "http://localhost:8002",
        valhalla_costing: str = "bus",
    ):
        self.enrich_osm = enrich_osm
        self.osm_poi_buffer_m = osm_poi_buffer_m
        self.osm_projected_crs = osm_projected_crs
        self.valhalla_url = valhalla_url
        self.valhalla_costing = valhalla_costing
        self._osm_extractor: Optional[OSMFeatureExtractor] = None
        
        if enrich_osm:
            self._osm_extractor = OSMFeatureExtractor(
                poi_buffer_m=osm_poi_buffer_m,
                projected_crs=osm_projected_crs,
                valhalla_url=valhalla_url,
                valhalla_costing=valhalla_costing,
            )

    def build(self, travel_times: pd.DataFrame) -> GraphArtifacts:
        """Build link graph from travel time observations."""
        nodes = self._build_link_nodes(travel_times)
        
        # Enrich with OSM features if enabled
        # In default mode: All features (road + POI) attached to link nodes
        # POI features use buffer around the route geometry (LineString)
        if self.enrich_osm and self._osm_extractor is not None:
            logger.info("Enriching nodes with OSM features...")
            osm_features = self._osm_extractor.extract_features(
                nodes,
                from_lon_col="from_lon",
                from_lat_col="from_lat",
                to_lon_col="to_lon",
                to_lat_col="to_lat",
            )
            # Merge OSM features into nodes
            nodes = pd.concat([nodes.reset_index(drop=True), osm_features.reset_index(drop=True)], axis=1)
            logger.info("Added %d OSM feature columns", len(osm_features.columns))
        
        _validate_nodes(nodes)

        edges = self._build_link_edges(travel_times, nodes)

        metadata: Dict[str, Any] = {
            "osm_enriched": self.enrich_osm,
            "osm_feature_columns": get_osm_feature_columns() if self.enrich_osm else [],
        }

        artifacts = GraphArtifacts(
            nodes=nodes,
            edges=edges,
            metadata=metadata,
        )

        return artifacts

    def _build_link_nodes(self, travel_df: pd.DataFrame) -> pd.DataFrame:
        """Build link nodes with geographic coordinates from unique from→to pairs.

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
            lambda row: f"{row['from_stop']}_{row['to_stop']}", axis=1
        )
        
        # Aggregate per-stop coordinates via geometric median of all observations.
        coord_map = _build_stop_coord_map(travel_df)
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

        return link_stats[['id', 'src', 'dst', 'from_lon', 'from_lat', 'to_lon', 'to_lat']]

    def _build_link_edges(self, travel_df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
        """Connect link nodes that share a stop: A→B when A ends and B starts there."""
        edges = []
        
        # Map stops to links ending/starting at that stop
        dst_to_links: Dict[str, list] = {}
        for _, node in nodes.iterrows():
            dst_stop = node['dst']
            if dst_stop not in dst_to_links:
                dst_to_links[dst_stop] = []
            dst_to_links[dst_stop].append(node['id'])
        
        src_to_links: Dict[str, list] = {}
        for _, node in nodes.iterrows():
            src_stop = node['src']
            if src_stop not in src_to_links:
                src_to_links[src_stop] = []
            src_to_links[src_stop].append(node['id'])
        
        # Connect: link ending at stop X → link starting at stop X
        for stop in dst_to_links:
            if stop in src_to_links:
                for link_ending in dst_to_links[stop]:
                    for link_starting in src_to_links[stop]:
                        if link_ending != link_starting:
                            edges.append({
                                'type': 'link_to_link',
                                'src': link_ending,
                                'dst': link_starting,
                            })
        
        cols = ['type', 'src', 'dst']
        if edges:
            edges_df = pd.DataFrame(edges)[cols]
        else:
            edges_df = pd.DataFrame(columns=cols)

        assert list(edges_df.columns) == cols, f"Expected columns {cols}, got {list(edges_df.columns)}"
        return edges_df


class BipartiteGraphBuilder(GraphBuilder):
    """Build a bipartite graph with separate link and stop nodes."""
    
    def __init__(
        self,
        enrich_osm: bool = False,
        osm_poi_buffer_m: float = 500.0,
        osm_projected_crs: str = "EPSG:28992",
        valhalla_url: str = "http://localhost:8002",
        valhalla_costing: str = "bus",
        gtfs_path: Optional[Path] = None,
        census_path: Optional[Path] = None,
    ):
        self.enrich_osm = enrich_osm
        self.osm_poi_buffer_m = osm_poi_buffer_m
        self.osm_projected_crs = osm_projected_crs
        self.valhalla_url = valhalla_url
        self.valhalla_costing = valhalla_costing
        self.gtfs_path = gtfs_path
        self.census_path = census_path
        self._osm_extractor: Optional[OSMFeatureExtractor] = None
        self._gtfs_extractor: Optional[GTFSFeatureExtractor] = None
        self._gisco_extractor: Optional[GISCOFeatureExtractor] = None
        self._landuse_extractor: Optional[LanduseFeatureExtractor] = None
        
        if enrich_osm:
            self._osm_extractor = OSMFeatureExtractor(
                poi_buffer_m=osm_poi_buffer_m,
                projected_crs=osm_projected_crs,
                valhalla_url=valhalla_url,
                valhalla_costing=valhalla_costing,
            )
            # Landuse extractor shares buffer/CRS with OSM extractor
            self._landuse_extractor = LanduseFeatureExtractor(
                buffer_m=osm_poi_buffer_m,
                projected_crs=osm_projected_crs,
            )
        
        if gtfs_path is not None:
            self._gtfs_extractor = GTFSFeatureExtractor(
                gtfs_path=gtfs_path,
            )
        
        if census_path is not None:
            self._gisco_extractor = GISCOFeatureExtractor(
                census_path=census_path,
            )

    def build(self, travel_times: pd.DataFrame) -> GraphArtifacts:
        """Build bipartite graph from travel time observations."""
        link_nodes = self._build_link_nodes(travel_times)
        
        # Build stop nodes
        stop_nodes = self._build_stop_nodes(travel_times)
        
        # Enrich with appropriate features if enabled
        # In bipartite mode: Road features attached to link nodes, POI features attached to stop nodes
        # POI features use buffer around the stop point (Point geometry)
        if self.enrich_osm and self._osm_extractor is not None:
            logger.info("Enriching link nodes with road features from Valhalla...")
            road_features = self._osm_extractor.extract_road_features(
                link_nodes,
                from_lon_col="from_lon",
                from_lat_col="from_lat",
                to_lon_col="to_lon",
                to_lat_col="to_lat",
            )
            # Merge road features into link nodes
            link_nodes = pd.concat([link_nodes.reset_index(drop=True), road_features.reset_index(drop=True)], axis=1)
            logger.info("Added %d road feature columns to link nodes", len(road_features.columns))
            
            logger.info("Enriching stop nodes with POI features from OSMnx...")
            poi_features = self._osm_extractor.extract_poi_features_for_stops(
                stop_nodes,
                lon_col="lon",
                lat_col="lat",
            )
            # Merge POI features into stop nodes
            stop_nodes = pd.concat([stop_nodes.reset_index(drop=True), poi_features.reset_index(drop=True)], axis=1)
            logger.info("Added %d POI feature columns to stop nodes", len(poi_features.columns))
        
        # Enrich stop nodes with landuse features if OSM enabled
        if self._landuse_extractor is not None:
            logger.info("Enriching stop nodes with landuse features from OSMnx...")
            landuse_features = self._landuse_extractor.extract_features(
                stop_nodes,
                lon_col="lon",
                lat_col="lat",
            )
            stop_nodes = pd.concat([stop_nodes.reset_index(drop=True), landuse_features.reset_index(drop=True)], axis=1)
            logger.info("Added %d landuse feature columns to stop nodes", len(landuse_features.columns))
        
        # Enrich stop nodes with GISCO census features if enabled
        if self._gisco_extractor is not None:
            logger.info("Enriching stop nodes with GISCO census features...")
            gisco_features = self._gisco_extractor.extract_features(
                stop_nodes,
                lon_col="lon",
                lat_col="lat",
            )
            stop_nodes = pd.concat([stop_nodes.reset_index(drop=True), gisco_features.reset_index(drop=True)], axis=1)
            logger.info("Added %d GISCO feature columns to stop nodes", len(gisco_features.columns))
        
        # Enrich stop nodes with GTFS features if enabled
        if self._gtfs_extractor is not None:
            logger.info("Enriching stop nodes with GTFS features...")
            gtfs_features = self._gtfs_extractor.extract_features(stop_nodes)
            # Merge GTFS features into stop nodes
            stop_nodes = pd.concat([stop_nodes.reset_index(drop=True), gtfs_features.reset_index(drop=True)], axis=1)
            logger.info("Added %d GTFS feature columns to stop nodes", len(gtfs_features.columns))
            
            logger.info("Enriching link nodes with GTFS features...")
            gtfs_link_features = self._gtfs_extractor.extract_link_features(link_nodes)
            # Merge GTFS features into link nodes
            link_nodes = pd.concat([link_nodes.reset_index(drop=True), gtfs_link_features.reset_index(drop=True)], axis=1)
            logger.info("Added %d GTFS feature columns to link nodes", len(gtfs_link_features.columns))
        
        _validate_nodes(link_nodes)
        _validate_nodes(stop_nodes)

        # Build edges (link-to-link connections)
        edges = self._build_bipartite_edges(travel_times, link_nodes, stop_nodes)

        metadata: Dict[str, Any] = {
            "osm_enriched": self.enrich_osm,
            "gtfs_enriched": self._gtfs_extractor is not None,
            "road_feature_columns": get_road_feature_columns() if self.enrich_osm else [],
            "poi_feature_columns": get_poi_feature_columns() if self.enrich_osm else [],
            "landuse_feature_columns": get_landuse_feature_columns() if self._landuse_extractor is not None else [],
            "gisco_feature_columns": get_gisco_feature_columns() if self._gisco_extractor is not None else [],
            "gtfs_feature_columns": get_gtfs_feature_columns() if self._gtfs_extractor is not None else [],
            "gtfs_link_feature_columns": get_gtfs_link_feature_columns() if self._gtfs_extractor is not None else [],
            "census_enriched": self._gisco_extractor is not None,
            "graph_mode": "bipartite",
        }

        # In bipartite mode, store stop nodes in the features field
        artifacts = GraphArtifacts(
            nodes=link_nodes,
            edges=edges,
            features=stop_nodes,  # Stop nodes stored here for bipartite graphs
            metadata=metadata,
        )

        return artifacts

    def _build_link_nodes(self, travel_df: pd.DataFrame) -> pd.DataFrame:
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
        
        # Aggregate per-stop coordinates via geometric median of all observations.
        coord_map = _build_stop_coord_map(travel_df)
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

    def _build_stop_nodes(self, travel_df: pd.DataFrame) -> pd.DataFrame:
        """Build stop nodes with coordinates aggregated by geometric median."""
        coord_map = _build_stop_coord_map(travel_df)
        all_stops = pd.DataFrame(
            [
                {'stop_id': stop_id, 'lon': lon, 'lat': lat}
                for stop_id, (lon, lat) in coord_map.items()
            ]
        )
        all_stops['id'] = all_stops['stop_id'].apply(lambda x: f"stop_{x}")
        all_stops['node_type'] = 'stop'
        return all_stops[['id', 'node_type', 'stop_id', 'lon', 'lat']]

    def _build_bipartite_edges(
        self,
        travel_df: pd.DataFrame,
        link_nodes: pd.DataFrame,
        stop_nodes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the 4 directed endpoint-aware edge types between stops and links."""
        edges = []

        for _, link in link_nodes.iterrows():
            src_stop = link['src']
            dst_stop = link['dst']
            link_id = link['id']

            edges.append({
                'type': 'stop_src_of_link',
                'src': f"stop_{src_stop}",
                'dst': link_id,
            })
            edges.append({
                'type': 'stop_dst_of_link',
                'src': f"stop_{dst_stop}",
                'dst': link_id,
            })
            edges.append({
                'type': 'link_to_src_stop',
                'src': link_id,
                'dst': f"stop_{src_stop}",
            })
            edges.append({
                'type': 'link_to_dst_stop',
                'src': link_id,
                'dst': f"stop_{dst_stop}",
            })
        
        cols = ['type', 'src', 'dst']
        if edges:
            edges_df = pd.DataFrame(edges)[cols]
        else:
            edges_df = pd.DataFrame(columns=cols)
        
        assert list(edges_df.columns) == cols, f"Expected columns {cols}, got {list(edges_df.columns)}"
        
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
