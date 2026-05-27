"""Base classes for graph construction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


class GraphBuilder(ABC):
    """Abstract base class for graph construction pipelines."""

    @abstractmethod
    def build(self, travel_times: pd.DataFrame) -> "GraphArtifacts":
        """Build graph artifacts from travel time data."""


@dataclass
class GraphArtifacts:
    """Container for graph construction outputs."""
    nodes: pd.DataFrame
    edges: pd.DataFrame
    features: Optional[pd.DataFrame] = None
    labels: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

