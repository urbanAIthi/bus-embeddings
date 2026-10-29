"""Inductive embeddings of public transport networks with heterogeneous GNNs."""

from importlib import metadata

try:
    __version__ = metadata.version("bus-embeddings")
except metadata.PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
