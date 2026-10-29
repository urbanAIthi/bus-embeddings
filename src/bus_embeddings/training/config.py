"""Experiment configuration: region split, data filters and model hyperparameters."""
from typing import Any, Dict, List

from bus_embeddings.features import FEATURE_GROUPS

# Global random seed
SEED = 42

# =============================================================================
# Region split
# =============================================================================
# Municipalities of the benchmark dataset by category.
MUNICIPALITY_CATEGORIES: Dict[str, List[str]] = {
    # Rotterdam, Den Haag, Utrecht
    "large_city": ["GM0599", "GM0518", "GM0344"],
    # Leiden, Delft, Nissewaard, Papendrecht, Midden-Delfland
    "small_city": ["GM0546", "GM0503", "GM1930", "GM0590", "GM1842"],
    # Tiel, Achtkarspelen, Veendam, Wassenaar, Bunnik
    "town": ["GM0281", "GM0059", "GM0047", "GM0629", "GM0312"],
    # Westerkwartier, Midden-Drenthe, Westerwolde, Borger-Odoorn, De Wolden
    "rural": ["GM1969", "GM1731", "GM1950", "GM1681", "GM1690"],
}

# Spatial split into training (A), validation (B) and test (C) regions,
# stratified by municipality category.
REGION_SPLIT: Dict[str, List[str]] = {
    "train": ["GM0281", "GM0503", "GM0546", "GM0590", "GM0599", "GM0629", "GM1681", "GM1950"],
    "val":   ["GM0047", "GM0059", "GM0518", "GM1731", "GM1842"],
    "test":  ["GM0312", "GM0344", "GM1690", "GM1930", "GM1969"],
}

# Temporal split: number of expanding-window folds, and the share of each
# held-out period used for validation (the remainder is the test period).
N_FOLDS = 5
VAL_FRACTION = 0.5

# =============================================================================
# Outlier filters (applied to all splits)
# =============================================================================
# Travel times are kept if MIN <= travel_time / ideal_travel_time <= MAX.
TRAVEL_TIME_FREEFLOW_RATIO_MIN: float = 0.8
TRAVEL_TIME_FREEFLOW_RATIO_MAX: float = 10.0

# Dwell times above this value (seconds) are removed.
DWELL_TIME_CAP: float = 120.0

# =============================================================================
# Targets and evaluation
# =============================================================================
# Link travel time target relative to the ideal travel time:
#   freeflow_ratio — travel_time / ideal_travel_time
#   freeflow       — travel_time - ideal_travel_time
TRAVEL_TIME_RESIDUAL_MODES = ("freeflow_ratio", "freeflow")

# Decision threshold of the stop classifier per model type, determined during
# development as the F1-optimal threshold on the validation regions averaged
# over the five folds (scripts/threshold_sweep.py) and kept fixed.
STOP_CLASSIFIER_THRESHOLDS: Dict[str, float] = {
    "graphsage": 0.40,
    "gat":       0.40,
    "xgboost":   0.33,
}

# =============================================================================
# Node features
# =============================================================================
FEATURE_CONFIG: Dict[str, bool] = {
    # Link features (OSM / Valhalla)
    "use-osm-length": True,
    "use-osm-travel-time": True,
    "use-osm-speed": True,
    "use-osm-class": True,
    "use-osm-lanes": True,
    "use-osm-traffic-lights": True,
    "use-osm-junctions": True,
    # Stop features: POI counts within 500 m (OSM)
    "use-osm-poi-food-drink": True,
    "use-osm-poi-shopping": True,
    "use-osm-poi-education": True,
    "use-osm-poi-healthcare": True,
    "use-osm-poi-transport": True,
    "use-osm-poi-finance": True,
    "use-osm-poi-entertainment": True,
    "use-osm-poi-sports-recreation": True,
    "use-osm-poi-tourism": True,
    "use-osm-poi-services": True,
    "use-osm-poi-worship": True,
    "use-osm-poi-other-amenity": True,
    # Stop features: land use within 500 m (OSM) and population density (Eurostat)
    "use-osm-landuse": True,
    "use-gisco-density": True,
    # Stop features (GTFS)
    "use-gtfs-poi-stops": True,
    "use-gtfs-trips": True,
    "use-gtfs-routes": True,
    "use-gtfs-headway": True,
    "use-gtfs-stop-pos": True,
    "use-gtfs-terminus": True,
    # Link features (GTFS)
    "use-gtfs-link-pos": True,
    "use-gtfs-link-frequency": True,
}

if set(FEATURE_CONFIG) != set(FEATURE_GROUPS):
    raise ValueError(
        "FEATURE_CONFIG and FEATURE_GROUPS are out of sync: "
        f"missing {set(FEATURE_GROUPS) - set(FEATURE_CONFIG)}, "
        f"extra {set(FEATURE_CONFIG) - set(FEATURE_GROUPS)}"
    )

# =============================================================================
# Model hyperparameters (tuned)
# =============================================================================
GRAPHSAGE_DEFAULTS: Dict[str, Any] = {
    # Encoder
    "gnn-hidden-dim": 48,
    "gnn-output-dim": 40,
    "gnn-layers": 2,
    "skip-connections": True,
    "global-skip-connection": True,
    "dropout": 0.21,
    "edge-dropout": 0.15,
    "feature-noise-std": 0.11,
    # Prediction heads
    "head-type": "film",
    "head-hidden-dim": 32,
    "n-head-layers": 2,
    # Optimisation
    "learning-rate": 1e-4,
    "gnn-lr-factor": 1.0,
    "weight-decay": 0.0025,
    "gnn-weight-decay": 0.0015,
    "grad-clip": 1.5,
    "epochs": 40,
    "early-stopping-patience": 12,
    "early-stopping-min-delta": 1e-5,
    "batch-size": 4096,
    "train-subsample": 0.1,
    "use-amp": True,
    "scheduler-factor": 0.62,
    "scheduler-patience": 4,
    "min-lr": 1e-6,
    # Targets and loss
    "travel-time-residual-mode": "freeflow_ratio",
    "log-transform": False,
    "huber-delta": 0.82,
    "focal-gamma": 2.0,
    "link-reg-weight": 2.0,
    "stop-reg-weight": 1.0,
    "stop-cls-weight": 1.05,
    **FEATURE_CONFIG,
}

GAT_DEFAULTS: Dict[str, Any] = {
    # Encoder
    "gnn-hidden-dim": 128,
    "gnn-output-dim": 32,
    "gnn-layers": 2,
    "num-heads": 4,
    "skip-connections": True,
    "global-skip-connection": True,
    "dropout": 0.2,
    "attn-dropout": 0.09,
    "edge-dropout": 0.05,
    "feature-noise-std": 0.035,
    # Prediction heads
    "head-type": "mlp",
    "head-hidden-dim": 32,
    "n-head-layers": 2,  # only used by the FiLM head
    # Optimisation
    "learning-rate": 7e-4,
    "gnn-lr-factor": 0.69,
    "weight-decay": 0.0021,
    "gnn-weight-decay": 0.0011,
    "grad-clip": 0.78,
    "epochs": 40,
    "early-stopping-patience": 12,
    "early-stopping-min-delta": 1e-5,
    "batch-size": 4096,
    "train-subsample": 0.1,
    "use-amp": True,
    "scheduler-factor": 0.31,
    "scheduler-patience": 4,
    "min-lr": 1e-6,
    # Targets and loss
    "travel-time-residual-mode": "freeflow_ratio",
    "log-transform": False,
    "huber-delta": 0.77,
    "focal-gamma": 2.0,
    "link-reg-weight": 2.0,
    "stop-reg-weight": 1.0,
    "stop-cls-weight": 1.0,
    **FEATURE_CONFIG,
}

# XGBoost baseline on the same node features and time context as the GNNs.
XGBOOST_DEFAULTS: Dict[str, Any] = {
    "xgb-n-estimators": 1000,
    "xgb-max-depth": 3,
    "xgb-learning-rate": 0.12,
    "xgb-subsample": 0.64,
    "xgb-colsample-bytree": 0.83,
    "xgb-min-child-weight": 29,
    "xgb-early-stopping-rounds": 20,
    "travel-time-residual-mode": "freeflow_ratio",
    "log-transform": False,
    # Only affects memory use during evaluation
    "batch-size": 16384,
    **FEATURE_CONFIG,
}

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "graphsage": GRAPHSAGE_DEFAULTS,
    "gat": GAT_DEFAULTS,
    "xgboost": XGBOOST_DEFAULTS,
}
