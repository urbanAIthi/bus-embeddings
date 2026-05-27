"""time2model: GNN-based transit travel time prediction with PyTorch Geometric."""

from importlib import metadata

try:
    __version__ = metadata.version("time2model")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
