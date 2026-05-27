"""Hyperparameter configuration for GNN embedding training."""
from typing import Any, Dict, List

from time2model.features import FEATURE_GROUPS

# Global random seed
SEED = 42

# Fixed train/val/test split: 8/5/5, stratified by category (see MUNICIPALITY_CATEGORIES in train_embeddings.py).
# GM0599 is pinned to train so train covers the largest graph-size regime.
# Remaining assignment via random.SystemRandom() on 2026-05-20.
REGION_SPLIT: Dict[str, List[str]] = {
    "train": ["GM0281", "GM0503", "GM0546", "GM0590", "GM0599", "GM0629", "GM1681", "GM1950"],
    "val":   ["GM0047", "GM0059", "GM0518", "GM1731", "GM1842"],
    "test":  ["GM0312", "GM0344", "GM1690", "GM1930", "GM1969"],
}

# Logging interval (epochs)
LOG_INTERVAL = 1

# Dataset configuration (from bus-graph-benchmark)
N_SPLITS = 5

# Outlier thresholds, applied to all splits
TRAVEL_TIME_FREEFLOW_RATIO_MIN: float = 0.8
TRAVEL_TIME_FREEFLOW_RATIO_MAX: float = 10.0
DWELL_TIME_CAP: float = 120.0

# Travel time residual mode:
#   historical_average — per-link, per-timeslot median
#   freeflow           — additive: travel_time − osm_travel_time
#   freeflow_ratio     — multiplicative: travel_time / osm_travel_time
#   none               — raw travel times
TRAVEL_TIME_RESIDUAL_MODE = "freeflow_ratio"
DWELL_TIME_RESIDUAL_MODE = "none"

# Stopped-vs-pass-through classification threshold on sigmoid.
# Values are the 5-split mean of the F1-optimal threshold on val
STOP_CLASSIFIER_THRESHOLDS: Dict[str, float] = {
    "graphsage": 0.40,
    "gat":       0.40,
    "xgboost":   0.33,
}

# Feature flags. All enabled except location.
FEATURE_CONFIG: Dict[str, bool] = {
    "use-location": False,
    # OSM link features (from Valhalla)
    "use-osm-length": True,
    "use-osm-travel-time": True,
    "use-osm-speed": True,
    "use-osm-class": True,
    "use-osm-lanes": True,
    "use-osm-traffic-lights": True,
    "use-osm-junctions": True,
    # OSM stop features (POI from OSMnx)
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
    # OSM landuse features
    "use-osm-landuse": True,
    # GISCO census features
    "use-gisco-density": True,
    # GTFS stop features
    "use-gtfs-poi-stops": True,
    "use-gtfs-trips": True,
    "use-gtfs-routes": True,
    "use-gtfs-headway": True,
    "use-gtfs-scheduled-dwell": False, # completely broken, mostly 0
    "use-gtfs-stop-pos": True,
    "use-gtfs-terminus": True,
    # GTFS link features
    "use-gtfs-link-travel-time": False, # completely broken, speeds up to 2000 km/h
    "use-gtfs-link-pos": True,
    "use-gtfs-link-frequency": True,
}

# Validate that all FEATURE_GROUPS have corresponding entries in the config
_all_flags = set(FEATURE_GROUPS.keys())
_config_flags = set(FEATURE_CONFIG.keys())
if _all_flags != _config_flags:
    _missing = _all_flags - _config_flags
    _extra = _config_flags - _all_flags
    raise ValueError(
        f"Feature config mismatch!\n"
        f"  Missing in FEATURE_CONFIG: {_missing}\n"
        f"  Extra in FEATURE_CONFIG: {_extra}"
    )


# Sourced from run: https://wandb.ai/urbanai/time2model-embeddings-graphsage/sweeps/w9neotx8/runs/pa18dlgg/overview?nw=nwuseralexhorn
EMBEDDING_GRAPHSAGE_DEFAULTS: Dict[str, Any] = {
    "gnn-hidden-dim": 48,
    "gnn-output-dim": 40,
    "gnn-layers": 2,
    "skip-connections": True,
    "global-skip-connection": True,
    "dropout": 0.21,
    "edge-dropout": 0.15,
    "sage-aggr": "mean",
    "add-route-shared-edges": False,
    "head-hidden-dim": 32,
    "head-type": "film",
    "head-time-concat": True,
    "n-head-layers": 2,
    "learning-rate": 1e-4,
    "gnn-lr-factor": 1.0,
    "weight-decay": 0.0025,
    "grad-clip": 1.5,
    "epochs": 40,
    "early-stopping-patience": 12,
    "early-stopping-min-delta": 1e-5,
    "batch-size": 4096,
    "use-amp": True,
    "num-workers": 8,
    "train-subsample": 0.1,
    "scheduler-type": "plateau",
    "scheduler-factor": 0.62,
    "scheduler-patience": 4,
    "min-lr": 1e-6,
    "log-transform": False,
    "travel-time-residual-mode": TRAVEL_TIME_RESIDUAL_MODE,
    "dwell-time-residual-mode": DWELL_TIME_RESIDUAL_MODE,
    "loss-type": "huber",
    "huber-delta": 0.82,
    "focal-gamma": 2.0,
    "link-reg-weight": 2.0,
    "stop-reg-weight": 1.0,
    "stop-cls-weight": 1.05,
    "feature-noise-std": 0.11,
    "gnn-weight-decay": 0.0015,
    **FEATURE_CONFIG,
}

# Sourced from run: https://wandb.ai/urbanai/time2model-embeddings-gat/sweeps/puugy49u/runs/yytyq97m/overview?nw=nwuseralexhorn
EMBEDDING_GAT_DEFAULTS: Dict[str, Any] = {
    "gnn-hidden-dim": 128,
    "gnn-output-dim": 32,
    "gnn-layers": 2,
    "skip-connections": True,
    "global-skip-connection": True,
    "dropout": 0.2,
    "attn-dropout": 0.09,
    "edge-dropout": 0.05,
    "num-heads": 4,
    "add-route-shared-edges": False,
    "head-hidden-dim": 32,
    "head-type": "mlp",
    "head-time-concat": True,
    "n-head-layers": 2,
    "learning-rate": 7e-4,
    "gnn-lr-factor": 0.69,
    "weight-decay": 0.0021,
    "gnn-weight-decay": 0.0011,
    "grad-clip": 0.78,
    "epochs": 40,
    "early-stopping-patience": 12,
    "early-stopping-min-delta": 1e-5,
    "batch-size": 4096,
    "use-amp": True,
    "num-workers": 8,
    "train-subsample": 0.1,
    "scheduler-type": "plateau",
    "scheduler-factor": 0.31,
    "scheduler-patience": 4,
    "min-lr": 1e-6,
    "log-transform": False,
    "travel-time-residual-mode": TRAVEL_TIME_RESIDUAL_MODE,
    "dwell-time-residual-mode": DWELL_TIME_RESIDUAL_MODE,
    "loss-type": "huber",
    "huber-delta": 0.77,
    "focal-gamma": 2.0,
    "link-reg-weight": 2.0,
    "stop-reg-weight": 1.0,
    "stop-cls-weight": 1.0,
    "feature-noise-std": 0.035,
    **FEATURE_CONFIG,
}

# Flat XGBoost baseline using the same dataset and features as the GNNs.
EMBEDDING_XGBOOST_DEFAULTS: Dict[str, Any] = {
    "xgb-n-estimators": 1000,
    "xgb-max-depth": 3,
    "xgb-learning-rate": 0.12,
    "xgb-subsample": 0.64,
    "xgb-colsample-bytree": 0.83,
    "xgb-min-child-weight": 29,
    "xgb-early-stopping-rounds": 20,
    "travel-time-residual-mode": TRAVEL_TIME_RESIDUAL_MODE,
    "dwell-time-residual-mode": DWELL_TIME_RESIDUAL_MODE,
    "log-transform": False,
    "train-subsample": 1.0,
    # XGBoost reads the dataset directly for fit(), but the shared eval path
    # iterates val/test loaders. Batch size only affects eval memory.
    "batch-size": 16384,
    "num-workers": 0,
    **FEATURE_CONFIG,
}

EMBEDDING_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "graphsage": "GraphSAGE-Embedding",
    "gat": "GAT-Embedding",
    "xgboost": "XGBoost-Baseline",
}
