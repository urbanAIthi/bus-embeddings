"""GTFS schedule-based feature extraction for stop and link nodes.

Stop features: gtfs_poi_stops_500m, gtfs_trips_1d, gtfs_routes_1d,
gtfs_headway_1d, gtfs_scheduled_dwell_time, gtfs_stop_pos, gtfs_is_terminus.

Link features: gtfs_scheduled_travel_time, gtfs_link_pos, gtfs_frequency_1d.
"""
from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..logging import get_logger

logger = get_logger(__name__)

# GTFS feature column names for stop nodes
GTFS_FEATURE_COLUMNS = [
    "gtfs_poi_stops_500m",
    "gtfs_trips_1d",
    "gtfs_routes_1d",
    "gtfs_headway_1d",
    "gtfs_scheduled_dwell_time",
    "gtfs_stop_pos",
    "gtfs_is_terminus",
]

# GTFS feature column names for link nodes
GTFS_LINK_FEATURE_COLUMNS = [
    "gtfs_scheduled_travel_time",
    "gtfs_link_pos",
    "gtfs_frequency_1d",
]


def get_gtfs_feature_columns() -> list[str]:
    """Return list of GTFS stop feature column names."""
    return GTFS_FEATURE_COLUMNS.copy()


def get_gtfs_link_feature_columns() -> list[str]:
    """Return list of GTFS link feature column names."""
    return GTFS_LINK_FEATURE_COLUMNS.copy()


def _time_str_to_seconds(time_str: str) -> Optional[float]:
    """Parse a GTFS time (``HH:MM:SS``, hours may exceed 24) to seconds since midnight."""
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) != 3:
            return None
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, TypeError, AttributeError):
        return None


class GTFSFeatureExtractor:
    """Extract GTFS-derived features for stop and link nodes.

    Stop features (trip count, headway, terminus, …) come from per-stop
    aggregates over ``stop_times.txt``; link features (scheduled travel
    time, frequency, …) come from consecutive stop pairs. Daily averages
    use ``calendar.txt`` when available; otherwise fall back to total
    trip counts.
    """

    def __init__(
        self,
        gtfs_path: Union[Path, str],
    ):
        self.gtfs_path = Path(gtfs_path)
        self._gtfs_data: Optional[dict[str, pd.DataFrame]] = None
        self._agg_features: Optional[pd.DataFrame] = None
        self._link_agg_features: Optional[pd.DataFrame] = None
        self._stop_code_to_gtfs: Optional[dict[str, list[str]]] = None
        self._service_days_per_week: Optional[dict[str, int]] = None
        self._gtfs_stop_coords: Optional[dict[str, tuple[float, float]]] = None

    def _load_gtfs_data(self) -> dict[str, pd.DataFrame]:
        """Load required GTFS files from zip archive."""
        if self._gtfs_data is not None:
            return self._gtfs_data

        if not self.gtfs_path.exists():
            raise FileNotFoundError(f"GTFS zip file not found: {self.gtfs_path}")

        # Required files
        files = {
            "agency": "agency.txt",
            "routes": "routes.txt",
            "trips": "trips.txt",
            "stops": "stops.txt",
            "stop_times": "stop_times.txt",
        }

        data = {}
        logger.info("Loading GTFS data from zip: %s", self.gtfs_path.name)

        with ZipFile(self.gtfs_path, "r") as zf:
            zip_files = zf.namelist()

            for name, filename in files.items():
                if filename not in zip_files:
                    raise FileNotFoundError(
                        f"Missing GTFS file in zip: {filename}"
                    )
                logger.debug("Loading %s from zip...", filename)
                with zf.open(filename) as f:
                    data[name] = pd.read_csv(
                        io.TextIOWrapper(f, encoding="utf-8"), dtype=str
                    )
                logger.debug("  Loaded %d rows", len(data[name]))

            # Optional: calendar.txt for computing service days
            if "calendar.txt" in zip_files:
                logger.debug("Loading calendar.txt from zip...")
                with zf.open("calendar.txt") as f:
                    data["calendar"] = pd.read_csv(
                        io.TextIOWrapper(f, encoding="utf-8"), dtype=str
                    )
                logger.debug("  Loaded %d rows", len(data["calendar"]))
            else:
                logger.info(
                    "calendar.txt not found in zip - will use total trip count "
                    "instead of daily averages"
                )

        self._gtfs_data = data
        return data

    def _build_stop_code_to_gtfs_id_map(
        self, gtfs_stops: pd.DataFrame
    ) -> dict[str, list[str]]:
        """Map stop_code → [stop_id, …] (one stop_code may cover several platforms)."""
        mapping: dict[str, list[str]] = defaultdict(list)
        for _, row in gtfs_stops.iterrows():
            stop_code = row.get("stop_code")
            stop_id = row["stop_id"]
            if pd.notna(stop_code) and stop_code:
                mapping[str(stop_code)].append(stop_id)
        return dict(mapping)

    def _build_stop_coords(
        self, gtfs_stops: pd.DataFrame
    ) -> dict[str, tuple[float, float]]:
        """Build mapping from stop_id to (lat, lon) coordinates."""
        coords: dict[str, tuple[float, float]] = {}
        for _, row in gtfs_stops.iterrows():
            stop_id = row["stop_id"]
            try:
                lat = float(row.get("stop_lat", 0))
                lon = float(row.get("stop_lon", 0))
                coords[stop_id] = (lat, lon)
            except (ValueError, TypeError):
                pass
        return coords

    def _compute_service_days_per_week(
        self, gtfs_data: dict[str, pd.DataFrame]
    ) -> dict[str, int]:
        """service_id → days/week (1–7) from ``calendar.txt``."""
        if "calendar" not in gtfs_data:
            return {}

        calendar = gtfs_data["calendar"]
        service_days = {}
        day_columns = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        for _, row in calendar.iterrows():
            service_id = row["service_id"]
            days_active = sum(
                int(row.get(day, 0)) for day in day_columns if day in row
            )
            service_days[service_id] = days_active

        logger.info(
            "Computed service days for %d service patterns", len(service_days)
        )
        return service_days

    def _compute_aggregated_features(self) -> pd.DataFrame:
        """Aggregate GTFS features per stop_id (DataFrame indexed by ``gtfs_stop_id``)."""
        if self._agg_features is not None:
            return self._agg_features

        gtfs_data = self._load_gtfs_data()

        trips_df = gtfs_data["trips"]
        trip_to_route = dict(zip(trips_df["trip_id"], trips_df["route_id"]))
        trip_to_service = dict(
            zip(
                trips_df["trip_id"],
                trips_df.get("service_id", [None] * len(trips_df)),
            )
        )
        logger.info(
            "Built trip->route mapping for %d trips", len(trip_to_route)
        )

        service_days_per_week = self._compute_service_days_per_week(gtfs_data)
        self._service_days_per_week = service_days_per_week

        # Process stop_times
        logger.debug("Processing stop_times...")
        stop_times = gtfs_data["stop_times"].copy()
        stop_times["stop_sequence"] = pd.to_numeric(
            stop_times["stop_sequence"], errors="coerce"
        )
        stop_times["route_id"] = stop_times["trip_id"].map(trip_to_route)
        stop_times["service_id"] = stop_times["trip_id"].map(trip_to_service)

        # Per-trip max sequence for terminus/position detection
        trip_max_seq = stop_times.groupby("trip_id")[
            "stop_sequence"
        ].transform("max")
        stop_times["max_seq"] = trip_max_seq

        stop_times["is_first"] = (stop_times["stop_sequence"] == 1).astype(int)
        stop_times["is_last"] = (
            stop_times["stop_sequence"] == stop_times["max_seq"]
        ).astype(int)

        # Normalized position in route (0.0 = start, 1.0 = end)
        stop_times["norm_position"] = (stop_times["stop_sequence"] - 1) / (
            stop_times["max_seq"] - 1
        ).replace(0, 1)
        stop_times["norm_position"] = stop_times["norm_position"].fillna(0.5)

        # Parse arrival/departure times for scheduled dwell computation
        stop_times["arrival_s"] = stop_times["arrival_time"].apply(
            _time_str_to_seconds
        )
        stop_times["departure_s"] = stop_times["departure_time"].apply(
            _time_str_to_seconds
        )
        # Scheduled dwell: departure - arrival at the same stop
        stop_times["dwell_s"] = (
            stop_times["departure_s"] - stop_times["arrival_s"]
        )
        # Filter out invalid dwell values
        stop_times.loc[
            stop_times["dwell_s"].isna() | (stop_times["dwell_s"] < 0),
            "dwell_s",
        ] = np.nan

        # Service days weight for daily averages
        if service_days_per_week:
            stop_times["service_days"] = (
                stop_times["service_id"]
                .map(service_days_per_week)
                .fillna(7)
                .astype(float)
            )
            stop_times["daily_weight"] = stop_times["service_days"] / 7.0
        else:
            stop_times["daily_weight"] = 1.0

        logger.debug("Aggregating features per GTFS stop...")

        daily_trips = (
            stop_times.groupby("stop_id")["daily_weight"]
            .sum()
            .rename("trips_1d")
        )

        # Distinct routes per day: sum daily weight per (stop, route), then count.
        stop_route_daily = (
            stop_times.groupby(["stop_id", "route_id"])["daily_weight"]
            .sum()
        )
        routes_1d = (
            stop_route_daily.groupby("stop_id")
            .apply(lambda x: len(x))
            .rename("routes_1d")
        )

        # Headway: mean seconds between consecutive departures per stop.
        sorted_deps = (
            stop_times.dropna(subset=["departure_s"])
            .sort_values(["stop_id", "departure_s"])
        )
        sorted_deps["dep_diff"] = sorted_deps.groupby("stop_id")[
            "departure_s"
        ].diff()
        headway_1d = (
            sorted_deps.groupby("stop_id")["dep_diff"]
            .mean()
            .rename("headway_1d")
        )

        dwell_time = (
            stop_times.groupby("stop_id")["dwell_s"]
            .mean()
            .rename("scheduled_dwell_time")
        )

        stop_pos = (
            stop_times.groupby("stop_id")["norm_position"]
            .mean()
            .rename("stop_pos")
        )

        # Terminus: stop is a terminus on a route iff first or last on any trip
        # of that route; flag the stop when this holds for >50% of its routes.
        stop_times["is_terminus_trip"] = (
            stop_times["is_first"] | stop_times["is_last"]
        ).astype(int)
        terminus_by_route = (
            stop_times.groupby(["stop_id", "route_id"])["is_terminus_trip"]
            .max()
        )
        terminus_route_count = (
            terminus_by_route.groupby("stop_id").sum().rename("terminus_routes")
        )
        total_route_count = (
            terminus_by_route.groupby("stop_id").count().rename("total_routes")
        )
        terminus_df = pd.concat(
            [terminus_route_count, total_route_count], axis=1
        )
        terminus_df["is_terminus"] = (
            terminus_df["terminus_routes"] > terminus_df["total_routes"] / 2
        ).astype(int)

        # Combine all features
        agg_features = pd.DataFrame(
            {
                "trips_1d": daily_trips,
                "routes_1d": routes_1d,
                "headway_1d": headway_1d,
                "scheduled_dwell_time": dwell_time,
                "stop_pos": stop_pos,
                "is_terminus": terminus_df["is_terminus"],
            }
        )
        agg_features.index.name = "gtfs_stop_id"
        agg_features = agg_features.fillna(
            {
                "trips_1d": 0.0,
                "routes_1d": 0,
                "headway_1d": 0.0,
                "scheduled_dwell_time": 0.0,
                "stop_pos": 0.5,
                "is_terminus": 0,
            }
        )

        logger.info("Computed features for %d GTFS stops", len(agg_features))

        self._agg_features = agg_features

        # Build stop_code mapping and stop coords
        self._stop_code_to_gtfs = self._build_stop_code_to_gtfs_id_map(
            gtfs_data["stops"]
        )
        self._gtfs_stop_coords = self._build_stop_coords(gtfs_data["stops"])
        logger.info(
            "Built mapping for %d stop codes", len(self._stop_code_to_gtfs)
        )

        return agg_features

    def _compute_link_aggregated_features(self) -> pd.DataFrame:
        """Per-link GTFS features computed over consecutive stop pairs.

        Columns: ``from_stop_id``, ``to_stop_id``, ``scheduled_travel_time``
        (mean seconds), ``link_pos`` (mean normalized 0–1 position),
        ``frequency_1d`` (average daily trips).
        """
        if self._link_agg_features is not None:
            return self._link_agg_features

        gtfs_data = self._load_gtfs_data()
        stop_times = gtfs_data["stop_times"].copy()

        stop_times["stop_sequence"] = pd.to_numeric(
            stop_times["stop_sequence"], errors="coerce"
        )
        stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

        # Parse times
        stop_times["departure_s"] = stop_times["departure_time"].apply(
            _time_str_to_seconds
        )
        stop_times["arrival_s"] = stop_times["arrival_time"].apply(
            _time_str_to_seconds
        )

        # Per-trip max sequence for normalizing position
        stop_times["max_seq"] = stop_times.groupby("trip_id")[
            "stop_sequence"
        ].transform("max")

        # Service days weight
        trips_df = gtfs_data["trips"]
        trip_to_service = dict(
            zip(
                trips_df["trip_id"],
                trips_df.get("service_id", [None] * len(trips_df)),
            )
        )

        service_days_per_week = self._service_days_per_week
        if service_days_per_week is None:
            service_days_per_week = self._compute_service_days_per_week(
                gtfs_data
            )
            self._service_days_per_week = service_days_per_week

        stop_times["service_id"] = stop_times["trip_id"].map(trip_to_service)
        if service_days_per_week:
            stop_times["daily_weight"] = (
                stop_times["service_id"]
                .map(service_days_per_week)
                .fillna(7)
                .astype(float)
                / 7.0
            )
        else:
            stop_times["daily_weight"] = 1.0

        # Create consecutive stop pairs by shifting within each trip
        stop_times["next_stop_id"] = stop_times.groupby("trip_id")[
            "stop_id"
        ].shift(-1)
        stop_times["next_arrival_s"] = stop_times.groupby("trip_id")[
            "arrival_s"
        ].shift(-1)

        # Drop last stop in each trip (no next stop)
        pairs = stop_times.dropna(subset=["next_stop_id"]).copy()

        # Scheduled travel time = arrival at next stop - departure at current
        pairs["scheduled_tt_s"] = (
            pairs["next_arrival_s"] - pairs["departure_s"]
        )
        # Handle negative values from data issues
        pairs.loc[pairs["scheduled_tt_s"] < 0, "scheduled_tt_s"] = np.nan

        # Normalized position of the link (use from_stop position)
        pairs["norm_position"] = (pairs["stop_sequence"] - 1) / (
            pairs["max_seq"] - 1
        ).replace(0, 1)
        pairs["norm_position"] = pairs["norm_position"].fillna(0.5)

        logger.info(
            "Aggregating GTFS features for %d link occurrences...", len(pairs)
        )

        # Aggregate per (from_stop, to_stop)
        link_agg = (
            pairs.groupby(["stop_id", "next_stop_id"])
            .agg(
                scheduled_travel_time=("scheduled_tt_s", "mean"),
                link_pos=("norm_position", "mean"),
                frequency_1d=("daily_weight", "sum"),
            )
            .reset_index()
        )
        link_agg = link_agg.rename(
            columns={
                "stop_id": "from_stop_id",
                "next_stop_id": "to_stop_id",
            }
        )
        link_agg["scheduled_travel_time"] = link_agg[
            "scheduled_travel_time"
        ].fillna(0.0)

        logger.info(
            "Computed link features for %d unique GTFS stop pairs",
            len(link_agg),
        )
        self._link_agg_features = link_agg
        return link_agg

    def _count_nearby_stops(
        self, stop_id: str, radius_m: float = 500.0
    ) -> int:
        """Count other GTFS stops within ``radius_m`` (Haversine; accurate enough at 500 m)."""
        if self._gtfs_stop_coords is None:
            return 0

        target = self._gtfs_stop_coords.get(stop_id)
        if target is None:
            return 0

        lat1, lon1 = target
        count = 0
        # Rough degree-to-meter conversion at Netherlands latitude (~52°N)
        # 1 degree lat ≈ 111,000m, 1 degree lon ≈ 111,000 * cos(52°) ≈ 68,300m
        lat_range = radius_m / 111_000.0
        lon_range = radius_m / 68_300.0

        for sid, (lat2, lon2) in self._gtfs_stop_coords.items():
            if sid == stop_id:
                continue
            # Quick bounding box filter
            if abs(lat2 - lat1) > lat_range or abs(lon2 - lon1) > lon_range:
                continue
            # Haversine distance
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(lat1))
                * np.cos(np.radians(lat2))
                * np.sin(dlon / 2) ** 2
            )
            dist_m = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            if dist_m <= radius_m:
                count += 1

        return count

    def _resolve_stop_code(self, raw_id: str) -> str:
        """Extract stop_code from a historical stop ID.

        Handles formats like ``OPERATOR:STOPCODE`` or plain ``STOPCODE``,
        and strips ``stop_`` prefixes.
        """
        s = str(raw_id)
        if s.startswith("stop_"):
            s = s[5:]
        if ":" in s:
            s = s.split(":", 1)[1]
        return s

    def extract_features(self, stop_nodes: pd.DataFrame) -> pd.DataFrame:
        """Extract GTFS stop features (aligned with ``stop_nodes``).

        Resolves each input ``stop_id`` (``OPERATOR:STOPCODE`` or plain
        ``STOPCODE``) to a stop_code, then aggregates across matching GTFS
        stop_ids.
        """
        agg_features = self._compute_aggregated_features()

        if self._stop_code_to_gtfs is None:
            raise RuntimeError(
                "Stop code mapping not built - "
                "call _compute_aggregated_features first"
            )

        logger.info(
            "Mapping GTFS features to %d stop nodes...", len(stop_nodes)
        )

        features_list = []
        matched = 0
        unmatched = 0

        for _, row in stop_nodes.iterrows():
            stop_id_col = row.get("stop_id", row.get("id", ""))
            stop_code = self._resolve_stop_code(stop_id_col)
            gtfs_stop_ids = self._stop_code_to_gtfs.get(stop_code, [])

            if gtfs_stop_ids:
                trips = 0.0
                routes = 0.0
                headways: list[float] = []
                dwells: list[float] = []
                positions: list[float] = []
                is_terminus = False
                nearby_stops = 0

                for gsid in gtfs_stop_ids:
                    if gsid in agg_features.index:
                        feats = agg_features.loc[gsid]
                        trips += feats["trips_1d"]
                        routes = max(routes, feats["routes_1d"])
                        if feats["headway_1d"] > 0:
                            headways.append(feats["headway_1d"])
                        dwells.append(feats["scheduled_dwell_time"])
                        positions.append(feats["stop_pos"])
                        is_terminus = is_terminus or bool(feats["is_terminus"])

                    # Count nearby stops (use first matching GTFS ID)
                    ns = self._count_nearby_stops(gsid)
                    nearby_stops = max(nearby_stops, ns)

                features_list.append(
                    {
                        "gtfs_poi_stops_500m": nearby_stops,
                        "gtfs_trips_1d": trips,
                        "gtfs_routes_1d": routes,
                        "gtfs_headway_1d": (
                            float(np.mean(headways)) if headways else 0.0
                        ),
                        "gtfs_scheduled_dwell_time": (
                            float(np.mean(dwells)) if dwells else 0.0
                        ),
                        "gtfs_stop_pos": (
                            float(np.mean(positions)) if positions else 0.5
                        ),
                        "gtfs_is_terminus": int(is_terminus),
                    }
                )
                matched += 1
            else:
                features_list.append(
                    {
                        "gtfs_poi_stops_500m": 0,
                        "gtfs_trips_1d": 0.0,
                        "gtfs_routes_1d": 0.0,
                        "gtfs_headway_1d": 0.0,
                        "gtfs_scheduled_dwell_time": 0.0,
                        "gtfs_stop_pos": 0.5,
                        "gtfs_is_terminus": 0,
                    }
                )
                unmatched += 1

        logger.info(
            "GTFS stop feature matching: %d matched, %d unmatched",
            matched,
            unmatched,
        )
        return pd.DataFrame(features_list)

    def extract_link_features(
        self, link_nodes: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract GTFS link features by looking up each (src, dst) pair in
        the consecutive stop pairs from ``stop_times.txt``.
        """
        # Ensure stop-level features are computed (builds stop_code mapping)
        self._compute_aggregated_features()
        link_agg = self._compute_link_aggregated_features()

        if self._stop_code_to_gtfs is None:
            raise RuntimeError(
                "Stop code mapping not built - "
                "call _compute_aggregated_features first"
            )

        # Build lookup dict keyed by (from_stop_id, to_stop_id)
        link_lookup: dict[tuple[str, str], dict[str, float]] = {}
        for _, r in link_agg.iterrows():
            key = (r["from_stop_id"], r["to_stop_id"])
            link_lookup[key] = {
                "scheduled_travel_time": r["scheduled_travel_time"],
                "link_pos": r["link_pos"],
                "frequency_1d": r["frequency_1d"],
            }

        logger.info(
            "Mapping GTFS features to %d link nodes...", len(link_nodes)
        )

        features_list = []
        matched = 0
        unmatched = 0

        for idx, row in link_nodes.iterrows():
            src_code = self._resolve_stop_code(row["src"])
            dst_code = self._resolve_stop_code(row["dst"])

            src_gtfs_ids = self._stop_code_to_gtfs.get(src_code, [])
            dst_gtfs_ids = self._stop_code_to_gtfs.get(dst_code, [])

            # Find all matching GTFS link entries (cross product of platforms)
            tt_values: list[float] = []
            pos_values: list[float] = []
            freq_total = 0.0

            for from_id in src_gtfs_ids:
                for to_id in dst_gtfs_ids:
                    feats = link_lookup.get((from_id, to_id))
                    if feats is not None:
                        tt_values.append(feats["scheduled_travel_time"])
                        pos_values.append(feats["link_pos"])
                        freq_total += feats["frequency_1d"]

            if tt_values:
                avg_tt = float(np.mean(tt_values))
                avg_pos = float(np.mean(pos_values))

                features_list.append(
                    {
                        "gtfs_scheduled_travel_time": avg_tt,
                        "gtfs_link_pos": avg_pos,
                        "gtfs_frequency_1d": freq_total,
                    }
                )
                matched += 1
            else:
                features_list.append(
                    {
                        "gtfs_scheduled_travel_time": 0.0,
                        "gtfs_link_pos": 0.5,
                        "gtfs_frequency_1d": 0.0,
                    }
                )
                unmatched += 1

        logger.info(
            "GTFS link feature matching: %d matched, %d unmatched",
            matched,
            unmatched,
        )
        return pd.DataFrame(features_list)
