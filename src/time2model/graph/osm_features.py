"""OpenStreetMap and Valhalla feature extraction for link and stop enrichment.

Link features from Valhalla routing: osm_length, osm_travel_time, osm_speed,
osm_class_*_frac (8 road classes), osm_lanes_*_frac (1-8 lanes),
osm_traffic_lights, osm_junctions (densities per km).

Stop features from OSMnx: osm_poi_*_500m (12 amenity categories within 500m buffer).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
from pathlib import Path
from shapely.geometry import LineString, MultiLineString

from ..logging import get_logger

logger = get_logger(__name__)

# Suppress FutureWarnings from osmnx/geopandas
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up osmnx cache directory
_osmnx_cache_dir = Path.home() / ".time2model-osmnx-cache"
_osmnx_cache_dir.mkdir(parents=True, exist_ok=True)
ox.settings.cache_folder = str(_osmnx_cache_dir)
ox.settings.log_console = True
ox.settings.use_cache = True

# Valhalla road class categories (from Valhalla's RoadClass enum)
VALHALLA_ROAD_CLASSES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service_other",
]

# Lane count categories (treat as categorical: 1-8 lanes)
LANE_COUNT_CATEGORIES = list(range(1, 9))  # [1, 2, 3, 4, 5, 6, 7, 8]

# POI categories based on common OSM amenity/shop tags relevant for transit
# Maps category name -> list of OSM tag values that belong to this category
POI_CATEGORIES = {
    "food_drink": [
        # amenity
        "restaurant", "cafe", "fast_food", "bar", "pub", "food_court",
        "ice_cream", "biergarten",
    ],
    "shopping": [
        # shop (all shop tags go here, handled separately)
    ],
    "education": [
        # amenity
        "school", "university", "college", "kindergarten", "library",
        "language_school", "music_school", "driving_school",
    ],
    "healthcare": [
        # amenity
        "hospital", "clinic", "doctors", "dentist", "pharmacy",
        "veterinary", "nursing_home",
    ],
    "transport": [
        # amenity
        "bus_station", "parking", "fuel", "bicycle_parking",
        "car_rental", "car_wash", "taxi", "charging_station",
    ],
    "finance": [
        # amenity
        "bank", "atm", "bureau_de_change",
    ],
    "entertainment": [
        # amenity
        "cinema", "theatre", "nightclub", "casino", "arts_centre",
        "community_centre", "events_venue",
    ],
    "sports_recreation": [
        # amenity
        "swimming_pool", "sports_centre", "gym", "fitness_centre",
        "playground", "park",
    ],
    "tourism": [
        # amenity
        "hotel", "hostel", "guest_house", "motel", "museum",
        "attraction", "viewpoint",
    ],
    "services": [
        # amenity
        "post_office", "police", "fire_station", "townhall",
        "courthouse", "embassy", "social_facility",
    ],
    "worship": [
        # amenity
        "place_of_worship", "monastery",
    ],
    "other_amenity": [
        # catch-all for other amenities
    ],
}

# List of POI category names for feature columns
POI_CATEGORY_NAMES = list(POI_CATEGORIES.keys())


class OSMFeatureExtractor:
    """Extract static features for transit links using Valhalla and OSMnx.

    Road features (length, travel_time, speed, road_class, lane_count) are
    extracted directly from Valhalla's trace_attributes response. POI features
    are fetched from OSMnx using a buffer around the route geometry.
    """

    def __init__(
        self,
        poi_buffer_m: float = 500.0,
        projected_crs: str = "EPSG:28992",
        valhalla_url: str = "http://localhost:8002",
        valhalla_costing: str = "bus",
    ):
        """Initialize with Valhalla URL and POI buffer settings."""
        self.poi_buffer_m = poi_buffer_m
        self.projected_crs = projected_crs
        self.valhalla_url = valhalla_url
        self.valhalla_costing = valhalla_costing

        # Initialize Valhalla router (required)
        from .valhalla_router import ValhallaRouter, test_valhalla_connection

        if not test_valhalla_connection(valhalla_url):
            raise RuntimeError(
                f"Valhalla server not reachable at {valhalla_url}. "
                "Please ensure the Valhalla service is running and accessible."
            )

        self._valhalla_router = ValhallaRouter(
            base_url=valhalla_url,
            costing=valhalla_costing,
        )
        logger.info("Valhalla routing initialized at %s", valhalla_url)

    def extract_features(
        self,
        nodes_df: pd.DataFrame,
        from_lon_col: str = "from_lon",
        from_lat_col: str = "from_lat",
        to_lon_col: str = "to_lon",
        to_lat_col: str = "to_lat",
    ) -> pd.DataFrame:
        """Extract link features (road + POI) from Valhalla routing."""
        # Determine if we have from/to coordinates
        has_endpoints = (
            from_lon_col in nodes_df.columns and to_lon_col in nodes_df.columns
        )

        if not has_endpoints:
            logger.warning("Missing from/to coordinates, returning empty features")
            return self._empty_features(len(nodes_df))

        coords = nodes_df[[from_lon_col, from_lat_col, to_lon_col, to_lat_col]].values
        from_coords_list = [(c[0], c[1]) for c in coords]
        to_coords_list = [(c[2], c[3]) for c in coords]

        # Log warning about invalid coordinates (0, 0)
        invalid_count = sum(
            1
            for fc, tc in zip(from_coords_list, to_coords_list)
            if fc == (0.0, 0.0) or tc == (0.0, 0.0)
        )
        if invalid_count > 0:
            logger.warning(
                "%d links have invalid coordinates (0,0) and will receive default features",
                invalid_count,
            )

        # Extract features using Valhalla routing
        return self._extract_features_with_valhalla(
            nodes_df, from_coords_list, to_coords_list
        )

    def extract_road_features(
        self,
        nodes_df: pd.DataFrame,
        from_lon_col: str = "from_lon",
        from_lat_col: str = "from_lat",
        to_lon_col: str = "to_lon",
        to_lat_col: str = "to_lat",
    ) -> pd.DataFrame:
        """Extract road features only (for bipartite link nodes)."""
        has_endpoints = (
            from_lon_col in nodes_df.columns and to_lon_col in nodes_df.columns
        )

        if not has_endpoints:
            logger.warning("Missing from/to coordinates, returning empty road features")
            return self._empty_road_features(len(nodes_df))

        coords = nodes_df[
            [from_lon_col, from_lat_col, to_lon_col, to_lat_col]
        ].values
        from_coords_list = [(c[0], c[1]) for c in coords]
        to_coords_list = [(c[2], c[3]) for c in coords]

        # Get routes for all links
        links = list(zip(from_coords_list, to_coords_list))
        routes = self._valhalla_router.get_routes_batch(links, show_progress=True)

        # Extract road features from Valhalla edge attributes
        features = []
        route_geometries: List[Optional[LineString]] = []

        for route in routes:
            if route is not None and route.get("edge_attributes"):
                feat = self._aggregate_valhalla_features(route)
                # Store geometry for traffic infrastructure query
                if route.get("geometry") and len(route["geometry"]) >= 2:
                    route_geometries.append(LineString(route["geometry"]))
                else:
                    route_geometries.append(None)
            else:
                feat = self._default_features()
                route_geometries.append(None)
            features.append(feat)

        features_df = pd.DataFrame(features)
        logger.info(
            "Extracted %d road features for %d links",
            len(features_df.columns),
            len(nodes_df),
        )

        return features_df

    def extract_poi_features_for_stops(
        self,
        stops_df: pd.DataFrame,
        lon_col: str = "lon",
        lat_col: str = "lat",
    ) -> pd.DataFrame:
        """Extract POI features for stop nodes using point-based buffer."""
        if lon_col not in stops_df.columns or lat_col not in stops_df.columns:
            logger.warning("Missing coordinates, returning empty POI features")
            return self._empty_poi_features(len(stops_df))

        coords = stops_df[[lon_col, lat_col]].values
        stop_points = [(c[0], c[1]) for c in coords]

        # Initialize POI features
        poi_counts_by_stop: Dict[str, List[float]] = {
            cat: [] for cat in POI_CATEGORY_NAMES
        }

        # Create point geometries for all stops
        from shapely.geometry import Point

        stop_geoms = [Point(lon, lat) for lon, lat in stop_points]

        # Build combined geometry for single POI query
        valid_geoms = [g for g in stop_geoms if g is not None and not g.is_empty]

        if not valid_geoms:
            return self._empty_poi_features(len(stops_df))

        try:
            # Create combined bounding box for POI query (with buffer)
            from shapely.geometry import MultiPoint

            combined_geom = MultiPoint(valid_geoms)
            gdf_geom = gpd.GeoDataFrame(
                [{"geometry": combined_geom}], crs="EPSG:4326"
            )
            gdf_proj = gdf_geom.to_crs(self.projected_crs)
            gdf_buffered = gdf_proj.buffer(self.poi_buffer_m)
            gdf_back = gpd.GeoDataFrame(
                geometry=gdf_buffered, crs=self.projected_crs
            ).to_crs("EPSG:4326")
            combined_polygon = gdf_back.geometry.values[0].simplify(0.0001)

            # Fetch POIs from OSMnx
            poi_gdf = self._fetch_pois(combined_polygon)

            if poi_gdf is None or len(poi_gdf) == 0:
                return self._empty_poi_features(len(stops_df))

            # Classify each POI into a category
            poi_gdf = self._classify_pois(poi_gdf)

            logger.info(
                "Processing %d POIs for %d stops using spatial joins",
                len(poi_gdf),
                len(stops_df),
            )

            # Project POIs to projected CRS and extract centroids
            poi_gdf_proj = poi_gdf.to_crs(self.projected_crs)
            poi_gdf_proj["centroid"] = poi_gdf_proj.geometry.centroid
            poi_gdf_points = gpd.GeoDataFrame(
                {
                    "geometry": poi_gdf_proj["centroid"],
                    "poi_category": poi_gdf_proj["poi_category"],
                },
                crs=self.projected_crs,
            )

            # Calculate POI counts by category for each stop
            for idx, geom in enumerate(stop_geoms):
                if geom is None or geom.is_empty:
                    for cat in POI_CATEGORY_NAMES:
                        poi_counts_by_stop[cat].append(0.0)
                    continue

                # Create buffer around this specific stop (in projected CRS)
                gdf_stop = gpd.GeoDataFrame(
                    [{"geometry": geom}], crs="EPSG:4326"
                )
                gdf_stop_proj = gdf_stop.to_crs(self.projected_crs)
                stop_buffer_proj = gdf_stop_proj.buffer(self.poi_buffer_m)
                stop_buffer_gdf = gpd.GeoDataFrame(
                    geometry=stop_buffer_proj, crs=self.projected_crs
                )

                # Use spatial join to find POIs within buffer
                pois_in_buffer = gpd.sjoin(
                    poi_gdf_points,
                    stop_buffer_gdf,
                    how="inner",
                    predicate="within",
                )

                # Count POIs by category
                if len(pois_in_buffer) > 0:
                    category_counts = (
                        pois_in_buffer["poi_category"].value_counts().to_dict()
                    )
                else:
                    category_counts = {}

                for cat in POI_CATEGORY_NAMES:
                    poi_counts_by_stop[cat].append(
                        float(category_counts.get(cat, 0))
                    )

        except Exception as e:
            logger.warning("Error extracting POI features for stops: %s", e)
            return self._empty_poi_features(len(stops_df))

        features_df = pd.DataFrame(
            {
                f"osm_poi_{cat}_500m": poi_counts_by_stop[cat]
                for cat in POI_CATEGORY_NAMES
            }
        )
        logger.info(
            "Extracted %d POI features for %d stops",
            len(features_df.columns),
            len(stops_df),
        )

        return features_df

    def _extract_features_with_valhalla(
        self,
        nodes_df: pd.DataFrame,
        from_coords_list: List[Tuple[float, float]],
        to_coords_list: List[Tuple[float, float]],
    ) -> pd.DataFrame:
        """Extract road + POI features using Valhalla routing."""
        logger.debug("Extracting features using Valhalla routing...")

        # Get routes for all links
        links = list(zip(from_coords_list, to_coords_list))
        routes = self._valhalla_router.get_routes_batch(links, show_progress=True)

        # Extract features for each link from Valhalla edge attributes
        features = []
        route_geometries: List[Optional[LineString]] = []

        for i, route in enumerate(routes):
            if route is not None and route.get("edge_attributes"):
                feat = self._aggregate_valhalla_features(route)
                # Store geometry for POI query
                if route.get("geometry") and len(route["geometry"]) >= 2:
                    route_geometries.append(LineString(route["geometry"]))
                else:
                    route_geometries.append(None)
            else:
                feat = self._default_features()
                route_geometries.append(None)

            features.append(feat)

        features_df = pd.DataFrame(features)

        # Add POI features using route geometries
        features_df = self._add_poi_features(features_df, route_geometries)

        logger.info(
            "Extracted %d features for %d links using Valhalla routing",
            len(features_df.columns),
            len(nodes_df),
        )

        return features_df

    # Default speed limit assumed for edges without an OSM speed_limit tag.
    # 50 km/h is the standard Dutch urban speed limit (most bus routes are urban).
    _DEFAULT_SPEED_LIMIT_KMH = 50.0

    def _aggregate_valhalla_features(
        self, route: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate Valhalla edge attributes into link-level features.

        Uses OSM speed_limit tags (not Valhalla's bus costing duration) for
        freeflow travel time — bus costing is 1.3-1.7x slower than observed.
        """
        edge_attrs = route.get("edge_attributes", [])

        if not edge_attrs:
            return self._default_features()

        # Total route distance from Valhalla directions API (meters)
        total_distance_m = route.get("distance_m", 0.0)

        # Initialize aggregators for edge-level features
        total_length_km = 0.0
        freeflow_travel_time_s = 0.0  # Σ(length / speed_limit)
        road_class_lengths = {rc: 0.0 for rc in VALHALLA_ROAD_CLASSES}
        lane_count_lengths = {lc: 0.0 for lc in LANE_COUNT_CATEGORIES}
        traffic_signal_count = 0
        junction_count = 0

        for edge in edge_attrs:
            length_km = edge.get("length", 0.0)  # Valhalla returns length in km
            road_class = edge.get("road_class", "service_other")
            lane_count = edge.get("lane_count", 2)

            total_length_km += length_km

            # Compute freeflow travel time from speed limits.
            # Use OSM speed_limit where available; fall back to default 50 km/h.
            speed_limit = edge.get("speed_limit", 0)
            if speed_limit <= 0:
                speed_limit = self._DEFAULT_SPEED_LIMIT_KMH
            if length_km > 0:
                freeflow_travel_time_s += (length_km / speed_limit) * 3600.0

            # Accumulate road class lengths
            if road_class in road_class_lengths:
                road_class_lengths[road_class] += length_km
            else:
                road_class_lengths["service_other"] += length_km

            # Accumulate lane count lengths (clamp to 1-8)
            lane_count = max(1, min(8, lane_count))
            lane_count_lengths[lane_count] += length_km

            # Count traffic signals (from both edge and node attributes)
            if edge.get("traffic_signal", False):
                traffic_signal_count += 1
            if edge.get("node_traffic_signal", False):
                traffic_signal_count += 1

            # Count junctions (street intersections at end nodes)
            node_type = edge.get("node_type", "")
            if node_type == "street_intersection":
                junction_count += 1

        # Build feature dict
        features: Dict[str, Any] = {}

        # Total length in meters (use Valhalla's distance_m)
        features["osm_length"] = total_distance_m

        # Freeflow travel time from OSM speed limits (seconds)
        # NOT Valhalla's duration_s — see docstring for rationale.
        features["osm_travel_time"] = freeflow_travel_time_s

        # Valhalla's bus-costing estimated travel time (seconds).
        # Includes penalties for turns, signals, road class, etc.
        features["osm_estimated_travel_time"] = route.get("duration_s", 0.0)

        # Freeflow speed in km/h
        if freeflow_travel_time_s > 0:
            features["osm_speed"] = (total_distance_m / freeflow_travel_time_s) * 3.6
        else:
            features["osm_speed"] = self._DEFAULT_SPEED_LIMIT_KMH

        # Road class distribution (normalized by total length to get fractions)
        for rc in VALHALLA_ROAD_CLASSES:
            if total_length_km > 0:
                features[f"osm_class_{rc}_frac"] = (
                    road_class_lengths[rc] / total_length_km
                )
            else:
                features[f"osm_class_{rc}_frac"] = 0.0

        # Lane count distribution (normalized by total length to get fractions)
        for lc in LANE_COUNT_CATEGORIES:
            if total_length_km > 0:
                features[f"osm_lanes_{lc}_frac"] = (
                    lane_count_lengths[lc] / total_length_km
                )
            else:
                features[f"osm_lanes_{lc}_frac"] = 0.0

        # Traffic infrastructure counts (per km)
        if total_length_km > 0:
            features["osm_traffic_lights"] = (
                float(traffic_signal_count) / total_length_km
            )
            features["osm_junctions"] = (
                float(junction_count) / total_length_km
            )
        else:
            features["osm_traffic_lights"] = 0.0
            features["osm_junctions"] = 0.0

        return features

    def _add_poi_features(
        self,
        features_df: pd.DataFrame,
        route_geometries: List[Optional[LineString]],
    ) -> pd.DataFrame:
        """Add POI count features by category using buffered route geometries."""
        # Initialize all POI category columns with zeros
        for cat in POI_CATEGORY_NAMES:
            features_df[f"osm_poi_{cat}_500m"] = 0.0

        # Build combined geometry for POI query
        valid_geoms = [g for g in route_geometries if g is not None]

        if not valid_geoms:
            return features_df

        try:
            # Create combined geometry for single OSMnx query
            combined_geom = MultiLineString(valid_geoms)
            gdf_geom = gpd.GeoDataFrame(
                [{"geometry": combined_geom}], crs="EPSG:4326"
            )
            gdf_proj = gdf_geom.to_crs(self.projected_crs)
            gdf_buffered = gdf_proj.buffer(self.poi_buffer_m)
            gdf_back = gpd.GeoDataFrame(
                geometry=gdf_buffered, crs=self.projected_crs
            ).to_crs("EPSG:4326")
            combined_polygon = gdf_back.geometry.values[0].simplify(0.0001)

            # Fetch POIs from OSMnx
            poi_gdf = self._fetch_pois(combined_polygon)

            if poi_gdf is None or len(poi_gdf) == 0:
                return features_df

            # Classify each POI into a category
            poi_gdf = self._classify_pois(poi_gdf)

            logger.info(
                "Processing %d POIs for %d routes using spatial joins",
                len(poi_gdf),
                len(route_geometries),
            )

            # Project POIs to projected CRS and extract centroids
            poi_gdf_proj = poi_gdf.to_crs(self.projected_crs)
            poi_gdf_proj["centroid"] = poi_gdf_proj.geometry.centroid
            poi_gdf_points = gpd.GeoDataFrame(
                {
                    "geometry": poi_gdf_proj["centroid"],
                    "poi_category": poi_gdf_proj["poi_category"],
                },
                crs=self.projected_crs,
            )

            # Calculate POI counts by category for each link
            poi_counts_by_link: Dict[str, List[float]] = {
                cat: [] for cat in POI_CATEGORY_NAMES
            }

            for idx, geom in enumerate(route_geometries):
                if geom is None:
                    for cat in POI_CATEGORY_NAMES:
                        poi_counts_by_link[cat].append(0.0)
                    continue

                # Create buffer around this specific route (in projected CRS)
                gdf_route = gpd.GeoDataFrame(
                    [{"geometry": geom}], crs="EPSG:4326"
                )
                gdf_route_proj = gdf_route.to_crs(self.projected_crs)
                route_buffer_proj = gdf_route_proj.buffer(self.poi_buffer_m)
                route_buffer_gdf = gpd.GeoDataFrame(
                    geometry=route_buffer_proj, crs=self.projected_crs
                )

                # Use spatial join to find POIs within buffer
                pois_in_buffer = gpd.sjoin(
                    poi_gdf_points,
                    route_buffer_gdf,
                    how="inner",
                    predicate="within",
                )

                # Count POIs by category
                if len(pois_in_buffer) > 0:
                    category_counts = (
                        pois_in_buffer["poi_category"].value_counts().to_dict()
                    )
                else:
                    category_counts = {}

                for cat in POI_CATEGORY_NAMES:
                    poi_counts_by_link[cat].append(
                        float(category_counts.get(cat, 0))
                    )

            # Add counts to dataframe
            for cat in POI_CATEGORY_NAMES:
                features_df[f"osm_poi_{cat}_500m"] = poi_counts_by_link[cat]

        except Exception as e:
            logger.warning("Error extracting POI features: %s", e)
            # Columns are already initialized to 0.0

        return features_df

    def _classify_pois(
        self, poi_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Classify POIs into categories based on their amenity/shop tags."""
        # Build reverse lookup: amenity value -> category
        amenity_to_category = {}
        for cat, amenities in POI_CATEGORIES.items():
            for amenity in amenities:
                amenity_to_category[amenity] = cat

        def classify_row(row: pd.Series) -> str:
            # Check if it's a shop (all shops go to 'shopping')
            if "shop" in row.index and pd.notna(row.get("shop")):
                return "shopping"
            # Check amenity tag
            if "amenity" in row.index and pd.notna(row.get("amenity")):
                amenity = str(row["amenity"]).lower()
                return amenity_to_category.get(amenity, "other_amenity")
            return "other_amenity"

        poi_gdf = poi_gdf.copy()
        poi_gdf["poi_category"] = poi_gdf.apply(classify_row, axis=1)

        # Log category distribution
        category_counts = poi_gdf["poi_category"].value_counts()
        logger.debug("POI category distribution: %s", category_counts.to_dict())

        return poi_gdf

    def _fetch_pois(self, polygon: Any) -> Optional[gpd.GeoDataFrame]:
        """Fetch POIs from OSMnx within the given polygon."""
        try:
            tags: Dict[str, Union[bool, str, List[str]]] = {
                "amenity": True,
                "shop": True,
            }
            gdf = ox.features_from_polygon(polygon=polygon, tags=tags)
            return gdf
        except Exception as e:
            logger.debug("Could not fetch POIs: %s", e)
            return None

    def _default_features(self) -> Dict[str, float]:
        """Return default features when route is not available."""
        features: Dict[str, float] = {
            "osm_length": 0.0,
            "osm_travel_time": 0.0,
            "osm_estimated_travel_time": 0.0,
            "osm_speed": 50.0,
        }

        # Road class distribution (all zeros)
        for rc in VALHALLA_ROAD_CLASSES:
            features[f"osm_class_{rc}_frac"] = 0.0

        # Lane count distribution (all zeros)
        for lc in LANE_COUNT_CATEGORIES:
            features[f"osm_lanes_{lc}_frac"] = 0.0

        # Traffic infrastructure counts (per km)
        features["osm_traffic_lights"] = 0.0
        features["osm_junctions"] = 0.0

        # Note: POI features are added separately in _add_poi_features

        return features

    def _empty_features(self, n: int) -> pd.DataFrame:
        """Return DataFrame with empty/default features."""
        features: Dict[str, List[float]] = {
            "osm_length": [0.0] * n,
            "osm_travel_time": [0.0] * n,
            "osm_speed": [50.0] * n,
        }

        # Road class distribution
        for rc in VALHALLA_ROAD_CLASSES:
            features[f"osm_class_{rc}_frac"] = [0.0] * n

        # Lane count distribution
        for lc in LANE_COUNT_CATEGORIES:
            features[f"osm_lanes_{lc}_frac"] = [0.0] * n

        # Traffic infrastructure counts (per km)
        features["osm_traffic_lights"] = [0.0] * n
        features["osm_junctions"] = [0.0] * n

        # POI counts by category
        for cat in POI_CATEGORY_NAMES:
            features[f"osm_poi_{cat}_500m"] = [0.0] * n

        return pd.DataFrame(features)

    def _empty_road_features(self, n: int) -> pd.DataFrame:
        """Return DataFrame with empty/default road features only."""
        features: Dict[str, List[float]] = {
            "osm_length": [0.0] * n,
            "osm_travel_time": [0.0] * n,
            "osm_speed": [50.0] * n,
        }

        # Road class distribution
        for rc in VALHALLA_ROAD_CLASSES:
            features[f"osm_class_{rc}_frac"] = [0.0] * n

        # Lane count distribution
        for lc in LANE_COUNT_CATEGORIES:
            features[f"osm_lanes_{lc}_frac"] = [0.0] * n

        # Traffic infrastructure counts (per km)
        features["osm_traffic_lights"] = [0.0] * n
        features["osm_junctions"] = [0.0] * n

        return pd.DataFrame(features)

    def _empty_poi_features(self, n: int) -> pd.DataFrame:
        """Return DataFrame with empty/default POI features only."""
        features: Dict[str, List[float]] = {}
        for cat in POI_CATEGORY_NAMES:
            features[f"osm_poi_{cat}_500m"] = [0.0] * n
        return pd.DataFrame(features)


def get_osm_feature_columns() -> List[str]:
    """Get the list of all OSM feature column names.

    Returns feature columns in the following groups:
    - Road features: osm_length, osm_travel_time, osm_speed
    - Road class distribution: osm_class_{class}_frac (8 classes)
    - Lane count distribution: osm_lanes_{n}_frac (8 categories)
    - Traffic infrastructure: osm_traffic_lights, osm_junctions
    - POI counts by category: osm_poi_{category}_500m (12 categories)

    Total: 3 + 8 + 8 + 2 + 12 = 33 features
    """
    cols = [
        "osm_length",
        "osm_travel_time",
        "osm_speed",
    ]
    for rc in VALHALLA_ROAD_CLASSES:
        cols.append(f"osm_class_{rc}_frac")
    for lc in LANE_COUNT_CATEGORIES:
        cols.append(f"osm_lanes_{lc}_frac")
    cols.append("osm_traffic_lights")
    cols.append("osm_junctions")
    for cat in POI_CATEGORY_NAMES:
        cols.append(f"osm_poi_{cat}_500m")
    return cols


def get_road_feature_columns() -> List[str]:
    """Get the list of road feature column names (from Valhalla).

    Returns feature columns:
    - Road features: osm_length, osm_travel_time, osm_speed
    - Road class distribution: osm_class_{class}_frac (8 classes)
    - Lane count distribution: osm_lanes_{n}_frac (8 categories)
    - Traffic infrastructure: osm_traffic_lights, osm_junctions

    Total: 3 + 8 + 8 + 2 = 21 features
    """
    cols = [
        "osm_length",
        "osm_travel_time",
        "osm_speed",
    ]
    for rc in VALHALLA_ROAD_CLASSES:
        cols.append(f"osm_class_{rc}_frac")
    for lc in LANE_COUNT_CATEGORIES:
        cols.append(f"osm_lanes_{lc}_frac")
    cols.append("osm_traffic_lights")
    cols.append("osm_junctions")
    return cols


def get_poi_feature_columns() -> List[str]:
    """Get the list of POI feature column names (from OSMnx).

    Returns feature columns:
    - POI counts by category: osm_poi_{category}_500m (12 categories)

    Total: 12 features
    """
    return [f"osm_poi_{cat}_500m" for cat in POI_CATEGORY_NAMES]
