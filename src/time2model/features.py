"""Node feature group definitions and configuration utilities.

Maps sweep hyperparameter flags (e.g. ``use-osm-length``) to CSV column names.
"""
from __future__ import annotations

from typing import Dict, List


# Valhalla road class categories
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

# Lane count categories (1-8)
LANE_COUNT_CATEGORIES = list(range(1, 9))

# POI categories (must match osm_features.py)
POI_CATEGORY_NAMES = [
    "food_drink",
    "shopping",
    "education",
    "healthcare",
    "transport",
    "finance",
    "entertainment",
    "sports_recreation",
    "tourism",
    "services",
    "worship",
    "other_amenity",
]

# Landuse categories (must match census_features.py)
LANDUSE_CATEGORY_NAMES = [
    "residential",
    "commercial",
    "industrial",
    "retail",
    "farmland",
    "forest",
    "recreation",
    "other",
]


# Define ALL feature groups with their column names
# Maps hyperparameter flag -> list of column names in CSV
FEATURE_GROUPS: Dict[str, List[str]] = {
    # Location / debug
    "use-location": [
        "from_lon",
        "from_lat",
        "to_lon",
        "to_lat",
    ],
    # OSM link features (from Valhalla)
    "use-osm-length": [
        "osm_length",
    ],
    "use-osm-travel-time": [
        "osm_travel_time",
    ],
    "use-osm-speed": [
        "osm_speed",
    ],
    "use-osm-class": [
        f"osm_class_{rc}_frac" for rc in VALHALLA_ROAD_CLASSES
    ],
    "use-osm-lanes": [
        f"osm_lanes_{lc}_frac" for lc in LANE_COUNT_CATEGORIES
    ],
    "use-osm-traffic-lights": [
        "osm_traffic_lights",
    ],
    "use-osm-junctions": [
        "osm_junctions",
    ],
    # OSM stop features (from OSMnx)
    "use-osm-poi-food-drink": ["osm_poi_food_drink_500m"],
    "use-osm-poi-shopping": ["osm_poi_shopping_500m"],
    "use-osm-poi-education": ["osm_poi_education_500m"],
    "use-osm-poi-healthcare": ["osm_poi_healthcare_500m"],
    "use-osm-poi-transport": ["osm_poi_transport_500m"],
    "use-osm-poi-finance": ["osm_poi_finance_500m"],
    "use-osm-poi-entertainment": ["osm_poi_entertainment_500m"],
    "use-osm-poi-sports-recreation": ["osm_poi_sports_recreation_500m"],
    "use-osm-poi-tourism": ["osm_poi_tourism_500m"],
    "use-osm-poi-services": ["osm_poi_services_500m"],
    "use-osm-poi-worship": ["osm_poi_worship_500m"],
    "use-osm-poi-other-amenity": ["osm_poi_other_amenity_500m"],
    # OSM landuse features
    "use-osm-landuse": [
        f"osm_landuse_{cat}" for cat in LANDUSE_CATEGORY_NAMES
    ],
    # GISCO census features
    "use-gisco-density": [
        "gisco_density",
    ],
    # GTFS stop features
    "use-gtfs-poi-stops": ["gtfs_poi_stops_500m"],
    "use-gtfs-trips": ["gtfs_trips_1d"],
    "use-gtfs-routes": ["gtfs_routes_1d"],
    "use-gtfs-headway": ["gtfs_headway_1d"],
    "use-gtfs-scheduled-dwell": ["gtfs_scheduled_dwell_time"],
    "use-gtfs-stop-pos": ["gtfs_stop_pos"],
    "use-gtfs-terminus": ["gtfs_is_terminus"],
    # GTFS link features
    "use-gtfs-link-travel-time": ["gtfs_scheduled_travel_time"],
    "use-gtfs-link-pos": ["gtfs_link_pos"],
    "use-gtfs-link-frequency": ["gtfs_frequency_1d"],
}


def get_enabled_columns(
    feature_config: Dict[str, bool],
    available_columns: List[str],
) -> List[str]:
    """Return columns enabled by ``feature_config`` that exist in ``available_columns``."""
    # Validate all required flags are present
    required_flags = set(FEATURE_GROUPS.keys())
    provided_flags = set(feature_config.keys())
    missing_flags = required_flags - provided_flags

    if missing_flags:
        raise ValueError(
            f"Feature config is missing required flags: {sorted(missing_flags)}\n"
            f"Make sure your config includes all flags from FEATURE_GROUPS.\n"
            f"Use FEATURE_CONFIG as a base: {{**FEATURE_CONFIG, 'use-location': True, ...}}"
        )

    enabled_cols = []
    available_set = set(available_columns)

    for flag, columns in FEATURE_GROUPS.items():
        if feature_config[flag]:  # No .get() fallback - fail if missing
            for col in columns:
                if col in available_set:
                    enabled_cols.append(col)

    return enabled_cols
