"""Valhalla road features for link nodes and OSMnx POI features for stop nodes.

Link features from Valhalla routing: osm_length, osm_travel_time (ideal time
from speed limits), osm_estimated_travel_time (Valhalla bus-costing estimate),
osm_speed, osm_class_*_frac (8 road classes), osm_lanes_*_frac (1-8 lanes),
osm_traffic_lights, osm_junctions (densities per km).

Stop features from OSMnx: osm_poi_*_500m (12 amenity categories within the
buffer around the stop).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
from osmnx._errors import InsufficientResponseError
from shapely.geometry import MultiPoint, Point

from bus_embeddings.graph.valhalla_router import ValhallaRouter, valhalla_is_available
from bus_embeddings.logging import get_logger

logger = get_logger(__name__)

# Suppress FutureWarnings from osmnx/geopandas
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up osmnx cache directory (also used for the landuse queries)
_osmnx_cache_dir = Path.home() / ".cache" / "bus-embeddings" / "osmnx"
_osmnx_cache_dir.mkdir(parents=True, exist_ok=True)
ox.settings.cache_folder = str(_osmnx_cache_dir)
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
    # Values of the leisure and tourism keys, which _fetch_pois does not
    # query: these two categories are always empty (see EXCLUDE_FEATURES).
    "sports_recreation": [
        "swimming_pool", "sports_centre", "gym", "fitness_centre",
        "playground", "park",
    ],
    "tourism": [
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
    """Extract static features for links (Valhalla) and stops (OSMnx).

    Road features (length, travel times, speed, road class, lane count, traffic
    lights, junctions) are aggregated from Valhalla's trace_attributes response
    along the bus route of each link. POI features are fetched from OSMnx using
    a buffer around each stop.
    """

    # Default speed limit assumed for edges without an OSM speed_limit tag.
    # 50 km/h is the standard Dutch urban speed limit (most bus routes are urban).
    _DEFAULT_SPEED_LIMIT_KMH = 50.0

    def __init__(
        self,
        poi_buffer_m: float = 500.0,
        projected_crs: str = "EPSG:28992",
        valhalla_url: str = "http://localhost:8002",
    ):
        """Initialize with Valhalla URL and POI buffer settings."""
        self.poi_buffer_m = poi_buffer_m
        self.projected_crs = projected_crs

        if not valhalla_is_available(valhalla_url):
            raise RuntimeError(
                f"Valhalla server not reachable at {valhalla_url}. "
                "Please ensure the Valhalla service is running and accessible."
            )

        self._valhalla_router = ValhallaRouter(base_url=valhalla_url)
        logger.info("Valhalla routing initialized at %s", valhalla_url)

    def extract_road_features(self, link_nodes: pd.DataFrame) -> pd.DataFrame:
        """Extract road features for link nodes (one Valhalla route per link)."""
        coords = link_nodes[["from_lon", "from_lat", "to_lon", "to_lat"]].values
        links = [((c[0], c[1]), (c[2], c[3])) for c in coords]
        routes = self._valhalla_router.get_routes_batch(links)

        # Links without a route or edge attributes get default features
        features = [
            self._aggregate_valhalla_features(route)
            if route is not None and route.get("edge_attributes")
            else self._default_features()
            for route in routes
        ]

        features_df = pd.DataFrame(features)
        logger.info(
            "Extracted %d road features for %d links",
            len(features_df.columns),
            len(link_nodes),
        )

        return features_df

    def extract_poi_features_for_stops(self, stop_nodes: pd.DataFrame) -> pd.DataFrame:
        """Extract POI counts per category within the buffer around each stop."""
        stop_geoms = [Point(lon, lat) for lon, lat in stop_nodes[["lon", "lat"]].values]

        poi_counts_by_stop: Dict[str, List[float]] = {
            cat: [] for cat in POI_CATEGORY_NAMES
        }

        # Buffered union of all stops, fetched with a single OSMnx query
        combined_geom = MultiPoint(stop_geoms)
        gdf_geom = gpd.GeoDataFrame(
            [{"geometry": combined_geom}], crs="EPSG:4326"
        )
        gdf_proj = gdf_geom.to_crs(self.projected_crs)
        gdf_buffered = gdf_proj.buffer(self.poi_buffer_m)
        gdf_back = gpd.GeoDataFrame(
            geometry=gdf_buffered, crs=self.projected_crs
        ).to_crs("EPSG:4326")
        combined_polygon = gdf_back.geometry.values[0].simplify(0.0001)

        poi_gdf = self._fetch_pois(combined_polygon)
        if poi_gdf is None:
            return self._empty_poi_features(len(stop_nodes))

        # Classify each POI into a category
        poi_gdf = self._classify_pois(poi_gdf)

        logger.info(
            "Processing %d POIs for %d stops using spatial joins",
            len(poi_gdf),
            len(stop_nodes),
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
        for geom in stop_geoms:
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

            category_counts = (
                pois_in_buffer["poi_category"].value_counts().to_dict()
            )
            for cat in POI_CATEGORY_NAMES:
                poi_counts_by_stop[cat].append(
                    float(category_counts.get(cat, 0))
                )


        features_df = pd.DataFrame(
            {
                f"osm_poi_{cat}_500m": poi_counts_by_stop[cat]
                for cat in POI_CATEGORY_NAMES
            }
        )
        logger.info(
            "Extracted %d POI features for %d stops",
            len(features_df.columns),
            len(stop_nodes),
        )

        return features_df

    def _aggregate_valhalla_features(
        self, route: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate Valhalla edge attributes into link-level features.

        Uses OSM speed_limit tags (not Valhalla's bus costing duration) for
        freeflow travel time — bus costing is 1.3-1.7x slower than observed.
        """
        # Total route distance from Valhalla directions API (meters)
        total_distance_m = route.get("distance_m", 0.0)

        # Initialize aggregators for edge-level features
        total_length_km = 0.0
        freeflow_travel_time_s = 0.0  # Σ(length / speed_limit)
        road_class_lengths = {rc: 0.0 for rc in VALHALLA_ROAD_CLASSES}
        lane_count_lengths = {lc: 0.0 for lc in LANE_COUNT_CATEGORIES}
        traffic_signal_count = 0
        junction_count = 0

        for edge in route["edge_attributes"]:
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
        """Fetch amenity/shop POIs within the polygon; None if there are none."""
        tags: Dict[str, Union[bool, str, List[str]]] = {
            "amenity": True,
            "shop": True,
        }
        try:
            return ox.features_from_polygon(polygon=polygon, tags=tags)
        except InsufficientResponseError:
            logger.warning("No POIs found in the area of the stops")
            return None

    def _default_features(self) -> Dict[str, float]:
        """Return default features when route is not available."""
        features: Dict[str, float] = {
            "osm_length": 0.0,
            "osm_travel_time": 0.0,
            "osm_estimated_travel_time": 0.0,
            "osm_speed": self._DEFAULT_SPEED_LIMIT_KMH,
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

        return features

    def _empty_poi_features(self, n: int) -> pd.DataFrame:
        """Return DataFrame with zero POI counts for ``n`` stops."""
        features: Dict[str, List[float]] = {}
        for cat in POI_CATEGORY_NAMES:
            features[f"osm_poi_{cat}_500m"] = [0.0] * n
        return pd.DataFrame(features)
