"""Valhalla routing client for fetching actual bus routes between stops."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import routingpy

from ..logging import get_logger

logger = get_logger(__name__)


class ValhallaRouter:
    """Route calculator using Valhalla via routingpy."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        costing: str = "bus",
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """Initialize with Valhalla server URL and caching settings."""
        self.base_url = base_url.rstrip("/")
        self.costing = costing
        self.use_cache = use_cache
        
        if cache_dir is None:
            cache_dir = Path.home() / ".time2model-valhalla-cache"
        self.cache_dir = Path(cache_dir)
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._client: Optional[Any] = None
        
        try:
            self._client = routingpy.Valhalla(base_url=self.base_url)
            logger.info("Initialized Valhalla router at %s with costing '%s'", self.base_url, self.costing)
        except Exception as e:
            logger.warning("Failed to initialize Valhalla client: %s", e)
            self._client = None
    
    def is_available(self) -> bool:
        """Check if the router is available."""
        return self._client is not None
    
    def get_route(
        self,
        from_coords: Tuple[float, float],
        to_coords: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        """Get route between two points, returning way IDs and geometry."""
        if self._client is None:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(from_coords, to_coords)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Make the trace_attributes request to get OSM way IDs
            route_result = self._request_route_with_ways(from_coords, to_coords)
            
            if route_result is not None:
                self._save_to_cache(cache_key, route_result)
                return route_result
            
            return None
            
        except Exception as e:
            logger.debug("Valhalla routing failed from %s to %s: %s", from_coords, to_coords, e)
            return None
    
    def _request_route_with_ways(
        self,
        from_coords: Tuple[float, float],
        to_coords: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        """Request route and extract OSM way IDs via trace_attributes."""
        import requests
        
        if self._client is None:
            return None
        
        # Fetch the basic route first to get the shape geometry.
        try:
            route = self._client.directions(
                locations=[[from_coords[0], from_coords[1]], [to_coords[0], to_coords[1]]],
                profile=self.costing,
            )
            
            if route is None or route.geometry is None:
                return None
            
            geometry = route.geometry
            distance_m = route.distance if route.distance else 0.0
            duration_s = route.duration if route.duration else 0.0
            
        except Exception as e:
            logger.debug("Failed to get basic route: %s", e)
            return None
        
        # Then trace_attributes to recover OSM way IDs along the route.
        try:
            # Build trace_attributes request
            # Shape points need to be in lat,lon format for Valhalla API
            shape_points = [{"lat": pt[1], "lon": pt[0]} for pt in geometry]
            
            # Subsample if too many points (Valhalla has limits)
            if len(shape_points) > 100:
                step = len(shape_points) // 100
                shape_points = shape_points[::step]
                if shape_points[-1] != {"lat": geometry[-1][1], "lon": geometry[-1][0]}:
                    shape_points.append({"lat": geometry[-1][1], "lon": geometry[-1][0]})
            
            trace_request = {
                "shape": shape_points,
                "costing": self.costing,
                "shape_match": "map_snap",
                "filters": {
                    "attributes": [
                        "edge.way_id",
                        "edge.length",
                        "edge.speed",
                        "edge.speed_limit",
                        "edge.road_class",
                        "edge.surface",
                        "edge.lane_count",
                        "edge.bridge",
                        "edge.tunnel",
                        "edge.traffic_signal",
                        "node.traffic_signal",
                        "node.type",
                    ],
                    "action": "include"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/trace_attributes",
                json=trace_request,
                timeout=30,
            )
            
            if response.status_code == 200:
                trace_data = response.json()
                way_ids = self._extract_way_ids(trace_data)
                edge_attributes = self._extract_edge_attributes(trace_data)
                
                return {
                    "way_ids": way_ids,
                    "geometry": geometry,
                    "distance_m": distance_m,
                    "duration_s": duration_s,
                    "edge_attributes": edge_attributes,
                }
            else:
                logger.debug(
                    "trace_attributes request failed with status %d: %s",
                    response.status_code,
                    response.text[:200] if response.text else "No response"
                )
                # Fall back to just returning geometry without way IDs
                return {
                    "way_ids": [],
                    "geometry": geometry,
                    "distance_m": distance_m,
                    "duration_s": duration_s,
                    "edge_attributes": [],
                }
                
        except Exception as e:
            logger.debug("trace_attributes request failed: %s", e)
            # Fall back to just returning geometry without way IDs
            return {
                "way_ids": [],
                "geometry": geometry,
                "distance_m": distance_m,
                "duration_s": duration_s,
                "edge_attributes": [],
            }
    
    def _extract_way_ids(self, trace_data: Dict[str, Any]) -> List[int]:
        """Extract unique OSM way IDs from trace_attributes response."""
        way_ids = []
        seen = set()
        
        edges = trace_data.get("edges", [])
        for edge in edges:
            way_id = edge.get("way_id")
            if way_id is not None and way_id not in seen:
                way_ids.append(way_id)
                seen.add(way_id)
        
        return way_ids
    
    def _extract_edge_attributes(self, trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract edge attributes from trace_attributes response."""
        edge_attrs = []
        
        edges = trace_data.get("edges", [])
        for edge in edges:
            # Get end_node info if available (contains node-level info like traffic signals)
            end_node = edge.get("end_node", {})
            node_type = end_node.get("type", "")
            node_traffic_signal = end_node.get("traffic_signal", False)
            
            attrs = {
                "way_id": edge.get("way_id"),
                "length": edge.get("length", 0),  # in km
                "speed": edge.get("speed", 0),  # in km/h (Valhalla costing speed)
                "speed_limit": edge.get("speed_limit", 0),  # in km/h (OSM speed limit tag)
                "road_class": edge.get("road_class", ""),
                "surface": edge.get("surface", ""),
                "lane_count": edge.get("lane_count", 0),
                "bridge": edge.get("bridge", False),
                "tunnel": edge.get("tunnel", False),
                "traffic_signal": edge.get("traffic_signal", False),
                "node_traffic_signal": node_traffic_signal,
                "node_type": node_type,  # e.g., 'street_intersection', 'gate', etc.
            }
            edge_attrs.append(attrs)
        
        return edge_attrs
    
    def get_routes_batch(
        self,
        links: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        show_progress: bool = True,
    ) -> List[Optional[Dict[str, Any]]]:
        """Get routes for multiple links with optional progress bar."""
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(links, desc="Fetching routes from Valhalla")
            except ImportError:
                iterator = links
        else:
            iterator = links
        
        for from_coords, to_coords in iterator:
            result = self.get_route(from_coords, to_coords)
            results.append(result)
        
        # Log statistics
        success_count = sum(1 for r in results if r is not None)
        with_ways_count = sum(1 for r in results if r is not None and len(r.get("way_ids", [])) > 0)
        logger.info(
            "Fetched %d/%d routes successfully (%d with way IDs)",
            success_count, len(links), with_ways_count
        )
        
        return results
    
    def _get_cache_key(
        self,
        from_coords: Tuple[float, float],
        to_coords: Tuple[float, float],
    ) -> str:
        """Generate cache key for a route request."""
        key_data = f"{from_coords[0]:.6f},{from_coords[1]:.6f}|{to_coords[0]:.6f},{to_coords[1]:.6f}|{self.costing}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load route result from cache."""
        if not self.use_cache:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                with cache_path.open("r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save route result to cache."""
        if not self.use_cache:
            return
        
        cache_path = self.cache_dir / f"{cache_key}.json"
        try:
            with cache_path.open("w") as f:
                json.dump(data, f)
        except OSError as e:
            logger.debug("Failed to save route to cache: %s", e)


def test_valhalla_connection(base_url: str = "http://localhost:8002") -> bool:
    """Return True if the Valhalla server responds at ``base_url``."""
    import requests

    base_url = base_url.rstrip("/")
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
