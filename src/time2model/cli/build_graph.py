"""
CLI for building graph artifacts from transit travel time data.

Generates nodes.csv, edges.csv, and metadata.json for use with the training pipeline.
Supports two modes: default (link-as-node) and bipartite (separate link + stop nodes).
See the graph/builder module and feature extractor modules for feature details.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..graph.builder import BipartiteGraphBuilder, LinkGraphBuilder
from ..logging import get_logger, setup_logging

logger = get_logger(__name__)

GRAPH_MODES = ["default", "bipartite"]


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
        "--enrich-osm",
        action="store_true",
        help="Enrich nodes with Valhalla route features and OSMnx POI data (requires Valhalla server)",
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
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        help="Filter to a single route by route ID (from the 'route' column in travel times)",
    )
    parser.add_argument(
        "--valhalla-url",
        type=str,
        default="http://localhost:8002",
        help="URL of the Valhalla routing service (default: http://localhost:8002, required for --enrich-osm)",
    )
    parser.add_argument(
        "--valhalla-costing",
        type=str,
        default="bus",
        choices=["bus", "auto", "truck", "bicycle", "pedestrian"],
        help="Valhalla costing profile for routing (default: bus)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=GRAPH_MODES,
        help="Graph construction mode: 'default' (link nodes only) or 'bipartite' (link + stop nodes)",
    )
    parser.add_argument(
        "--gtfs-zip",
        type=Path,
        default=None,
        help="Path to GTFS zip file. Daily departures computed from calendar.txt if available. Only used in bipartite mode.",
    )
    parser.add_argument(
        "--census-path",
        type=Path,
        default=None,
        help="Path to GISCO census geoparquet file (e.g. ESTAT_Census_2021_V2.parquet). Adds gisco_density feature. Only used in bipartite mode.",
    )
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)

    start_time = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    travel_times = pd.read_csv(args.travel_times, parse_dates=["from_time", "to_time"])
    travel_times['travel_time_s'] = (travel_times["to_time"] - travel_times["from_time"]).dt.total_seconds() # type: ignore
    
    if args.route is not None:
        original_count = len(travel_times)
        # Convert to int if route column is numeric
        route_filter = int(args.route) if args.route.isdigit() else args.route
        travel_times = travel_times[travel_times["route"] == route_filter]
        logger.info(
            "Filtered to route '%s': %d -> %d records",
            args.route,
            original_count,
            len(travel_times),
        )
        if travel_times.empty:
            logger.error("No records found for route '%s'", args.route)
            return
    
    # Select graph builder based on mode
    if args.mode == "bipartite":
        builder = BipartiteGraphBuilder(
            enrich_osm=args.enrich_osm,
            osm_poi_buffer_m=args.osm_poi_buffer,
            osm_projected_crs=args.osm_projected_crs,
            valhalla_url=args.valhalla_url,
            valhalla_costing=args.valhalla_costing,
            gtfs_path=args.gtfs_zip,
            census_path=args.census_path,
        )
    else:
        if args.gtfs_zip is not None:
            logger.warning("--gtfs-zip is only used in bipartite mode, ignoring")
        if args.census_path is not None:
            logger.warning("--census-path is only used in bipartite mode, ignoring")
        builder = LinkGraphBuilder(
            enrich_osm=args.enrich_osm,
            osm_poi_buffer_m=args.osm_poi_buffer,
            osm_projected_crs=args.osm_projected_crs,
            valhalla_url=args.valhalla_url,
            valhalla_costing=args.valhalla_costing,
        )
    logger.info("Using %s graph builder (mode: %s)", type(builder).__name__, args.mode)
    
    artifacts = builder.build(travel_times)

    # Write output files based on graph mode
    if args.mode == "bipartite":
        _write_csv(artifacts.nodes, args.output_dir / "nodes_link.csv")
        _write_csv(artifacts.edges, args.output_dir / "edges.csv")
        if artifacts.features is not None:
            _write_csv(artifacts.features, args.output_dir / "nodes_stop.csv")
    else:
        _write_csv(artifacts.nodes, args.output_dir / "nodes.csv")
        _write_csv(artifacts.edges, args.output_dir / "edges.csv")
    
    # Add mode to metadata
    artifacts.metadata["graph_mode"] = args.mode
    _write_json(artifacts.metadata, args.output_dir / "metadata.json")

    elapsed = time.time() - start_time
    logger.info("Graph artifacts written to %s in %.2fs", args.output_dir.resolve(), elapsed)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

if __name__ == "__main__":
    main()
