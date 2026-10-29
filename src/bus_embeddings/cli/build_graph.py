"""
CLI for building the bipartite graph of one region from transit travel time data.

Writes nodes_link.csv, nodes_stop.csv and edges.csv for use with the training
pipeline. Link nodes are enriched via Valhalla and GTFS, stop nodes via OSMnx,
the GISCO census grid and GTFS; see the graph.builder module and the feature
extractor modules for feature details.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from bus_embeddings.graph.builder import BipartiteGraphBuilder
from bus_embeddings.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Build graph artifacts from travel time data."""
    parser = argparse.ArgumentParser(
        description="Build graph artifacts from transit travel time data."
    )
    parser.add_argument(
        "--travel-times",
        type=Path,
        required=True,
        help="Path to CSV file containing travel time data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write artifacts into (default: ./artifacts).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level (default: INFO)",
    )
    parser.add_argument(
        "--gtfs-zip",
        type=Path,
        required=True,
        help="Path to GTFS zip file. Daily departures computed from calendar.txt if available.",
    )
    parser.add_argument(
        "--census-path",
        type=Path,
        required=True,
        help="Path to GISCO census geoparquet file (e.g. ESTAT_Census_2021_V2.parquet). Adds gisco_density feature.",
    )
    parser.add_argument(
        "--valhalla-url",
        type=str,
        default="http://localhost:8002",
        help="URL of the Valhalla routing service (default: http://localhost:8002)",
    )
    parser.add_argument(
        "--osm-poi-buffer",
        type=float,
        default=500.0,
        help="Buffer in meters around stops for POI/landuse queries (default: 500)",
    )
    parser.add_argument(
        "--osm-projected-crs",
        type=str,
        default="EPSG:28992",
        help="Projected CRS for accurate buffering in meters (default: EPSG:28992 for Netherlands)",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    start_time = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    travel_times = pd.read_csv(args.travel_times)

    builder = BipartiteGraphBuilder(
        gtfs_path=args.gtfs_zip,
        census_path=args.census_path,
        valhalla_url=args.valhalla_url,
        osm_poi_buffer_m=args.osm_poi_buffer,
        osm_projected_crs=args.osm_projected_crs,
    )
    artifacts = builder.build(travel_times)

    artifacts.link_nodes.to_csv(args.output_dir / "nodes_link.csv", index=False)
    artifacts.edges.to_csv(args.output_dir / "edges.csv", index=False)
    artifacts.stop_nodes.to_csv(args.output_dir / "nodes_stop.csv", index=False)

    elapsed = time.time() - start_time
    logger.info("Graph artifacts written to %s in %.2fs", args.output_dir.resolve(), elapsed)


if __name__ == "__main__":
    main()
