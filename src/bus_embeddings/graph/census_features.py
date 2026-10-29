"""GISCO Census population density and OSM landuse features for stop nodes.

Census: population density from EU Census 2021 1km grid (GISCO/Eurostat).
Landuse: fraction of the buffer around a stop covered by 8 landuse categories (OSMnx).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
from osmnx._errors import InsufficientResponseError
import pandas as pd
from shapely.geometry import MultiPoint, Point

from bus_embeddings.logging import get_logger

logger = get_logger(__name__)

# Landuse categories relevant for transit demand
# Maps category name -> list of OSM landuse values
LANDUSE_CATEGORIES: Dict[str, List[str]] = {
    "residential": [
        "residential",
    ],
    "commercial": [
        "commercial",
    ],
    "industrial": [
        "industrial",
        "construction",
        "quarry",
        "landfill",
        "port",
        "railway",
    ],
    "retail": [
        "retail",
    ],
    "farmland": [
        "farmland",
        "farmyard",
        "meadow",
        "orchard",
        "vineyard",
        "greenhouse_horticulture",
        "allotments",
    ],
    "forest": [
        "forest",
    ],
    "recreation": [
        "recreation_ground",
        "village_green",
        "grass",
        "park",
    ],
    # 'other' is the catch-all, not explicitly mapped
}

LANDUSE_CATEGORY_NAMES = list(LANDUSE_CATEGORIES.keys()) + ["other"]


class GISCOFeatureExtractor:
    """Extract population density from GISCO Census 2021 1km grid."""

    def __init__(self, census_path: Union[Path, str]):
        self.census_path = Path(census_path)

    def _load_grid(self) -> gpd.GeoDataFrame:
        """Load the populated, valid cells of the census grid."""
        if not self.census_path.exists():
            raise FileNotFoundError(
                f"Census file not found: {self.census_path}"
            )

        logger.info("Loading GISCO census grid from %s...", self.census_path.name)
        grid = gpd.read_parquet(self.census_path)
        logger.debug("Loaded %d grid cells (CRS: %s)", len(grid), grid.crs)

        # Use raw population count as density: each grid cell is 1 km², so T
        # is already persons/km². (Dividing by LAND_SURFACE creates extreme
        # outliers for coastal/river cells with tiny land slivers.)
        grid["T"] = pd.to_numeric(grid["T"], errors="coerce").fillna(0)
        grid["density"] = grid["T"]

        # Filter out unpopulated cells to speed up spatial join
        grid = grid[grid["POPULATED"] == 1].copy()
        logger.debug("Filtered to %d populated grid cells", len(grid))

        # Ensure valid geometry
        grid = grid[grid.geometry.is_valid & ~grid.geometry.is_empty].copy()

        return grid

    def extract_features(self, stop_nodes: pd.DataFrame) -> pd.DataFrame:
        """Extract population density for each stop (0 outside populated cells)."""
        grid = self._load_grid()

        # Create point GeoDataFrame from stop coords (WGS84)
        points = [
            Point(lon, lat)
            for lon, lat in zip(
                stop_nodes["lon"].values, stop_nodes["lat"].values, strict=True
            )
        ]
        stops_gdf = gpd.GeoDataFrame(
            {"geometry": points},
            crs="EPSG:4326",
            index=stop_nodes.index,
        )

        # Reproject stops to census CRS (EPSG:3035 - ETRS89/LAEA)
        stops_proj = stops_gdf.to_crs(grid.crs)

        logger.info(
            "Spatial join: %d stops against %d census cells...",
            len(stops_proj),
            len(grid),
        )

        # Spatial join: find which grid cell contains each stop
        joined = gpd.sjoin(
            stops_proj,
            grid,
            how="left",
            predicate="within",
        )

        # Handle duplicates (stop at cell boundary) — take first match
        joined = joined[~joined.index.duplicated(keep="first")]

        # Reindex to match original order
        density = joined["density"].reindex(stop_nodes.index).fillna(0.0)

        logger.info(
            "GISCO density: matched %d/%d stops (mean=%.1f, median=%.1f)",
            (density > 0).sum(),
            len(density),
            density.mean(),
            density.median(),
        )

        return pd.DataFrame({"gisco_density": density.values})


class LanduseFeatureExtractor:
    """Extract OSM landuse fractions within buffer around each stop."""

    def __init__(
        self,
        buffer_m: float = 500.0,
        projected_crs: str = "EPSG:28992",
    ):
        self.buffer_m = buffer_m
        self.projected_crs = projected_crs

    def extract_features(self, stop_nodes: pd.DataFrame) -> pd.DataFrame:
        """Extract landuse fractions for each stop."""
        # Build all stop points
        stop_points = [
            Point(lon, lat)
            for lon, lat in zip(
                stop_nodes["lon"].values, stop_nodes["lat"].values, strict=True
            )
        ]

        # Buffered union of all stops for a single OSMnx query
        combined_geom = MultiPoint(stop_points)
        gdf_geom = gpd.GeoDataFrame(
            [{"geometry": combined_geom}], crs="EPSG:4326"
        )
        gdf_proj = gdf_geom.to_crs(self.projected_crs)
        gdf_buffered = gdf_proj.buffer(self.buffer_m)
        gdf_back = gpd.GeoDataFrame(
            geometry=gdf_buffered, crs=self.projected_crs
        ).to_crs("EPSG:4326")
        combined_polygon = gdf_back.geometry.values[0].simplify(0.0001)

        try:
            landuse_gdf = ox.features_from_polygon(
                polygon=combined_polygon,
                tags={"landuse": True},
            )
        except InsufficientResponseError:
            logger.warning("No land use found in the area of the stops")
            return self._empty_features(len(stop_nodes))

        # Classify landuse
        landuse_gdf = self._classify_landuse(landuse_gdf)

        # Project for area calculation
        landuse_proj = landuse_gdf.to_crs(self.projected_crs)

        logger.info(
            "Processing %d landuse polygons for %d stops",
            len(landuse_proj),
            len(stop_nodes),
        )

        # Buffer area for normalization
        buffer_area = np.pi * self.buffer_m**2

        # Calculate landuse fractions for each stop
        results: Dict[str, List[float]] = {
            f"osm_landuse_{cat}": [] for cat in LANDUSE_CATEGORY_NAMES
        }

        for point in stop_points:
            # Create buffer around stop
            gdf_stop = gpd.GeoDataFrame(
                [{"geometry": point}], crs="EPSG:4326"
            )
            gdf_stop_proj = gdf_stop.to_crs(self.projected_crs)
            stop_buffer = gdf_stop_proj.geometry.values[0].buffer(self.buffer_m)

            # Intersect with landuse polygons
            for cat in LANDUSE_CATEGORY_NAMES:
                cat_landuse = landuse_proj[
                    landuse_proj["landuse_category"] == cat
                ]
                if len(cat_landuse) == 0:
                    results[f"osm_landuse_{cat}"].append(0.0)
                    continue

                # Compute intersection area
                try:
                    intersections = cat_landuse.geometry.intersection(
                        stop_buffer
                    )
                    total_area = intersections.area.sum()
                    frac = min(total_area / buffer_area, 1.0)
                    results[f"osm_landuse_{cat}"].append(frac)
                except Exception:
                    results[f"osm_landuse_{cat}"].append(0.0)

        logger.debug("Extracted landuse features for %d stops", len(stop_nodes))
        return pd.DataFrame(results)

    def _classify_landuse(
        self, landuse_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Classify landuse polygons into categories."""
        # Build reverse lookup
        tag_to_category: Dict[str, str] = {}
        for cat, tags in LANDUSE_CATEGORIES.items():
            for tag in tags:
                tag_to_category[tag] = cat

        landuse_gdf = landuse_gdf.copy()
        landuse_gdf["landuse_category"] = landuse_gdf["landuse"].apply(
            lambda x: tag_to_category.get(str(x).lower(), "other")
            if pd.notna(x)
            else "other"
        )

        # Log distribution
        dist = landuse_gdf["landuse_category"].value_counts()
        logger.debug("Landuse distribution: %s", dist.to_dict())

        return landuse_gdf

    def _empty_features(self, n: int) -> pd.DataFrame:
        """Return DataFrame with zero landuse fractions."""
        return pd.DataFrame(
            {
                f"osm_landuse_{cat}": [0.0] * n
                for cat in LANDUSE_CATEGORY_NAMES
            }
        )
