#!/usr/bin/env python3
"""Print Tables III-V and the embedding diversity (Sec. VI-B) of the paper.

Reads the ``metrics.json`` files that ``train_embeddings`` and
``evaluate_baselines`` write to ``<results-dir>/<run>/`` (see
scripts/reproduce_paper_tables.sh for the runs). All values are geometric
means over the five folds (arithmetic mean for MADGap). All values are
printed with four decimals; the paper rounds Tables IV and V to three.

Usage::

    python scripts/make_paper_tables.py --results-dir results
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# Table row -> results directory
MODELS = {
    "GraphSAGE + Linear": "graphsage",
    "GraphSAGE only": "graphsage_only",
    "GATv2 + Linear": "gat",
    "GATv2 only": "gat_only",
    "Linear": "linear",
    "XGBoost": "xgboost",
}
BASELINES = {
    "GTFS sched.": "gtfs_schedule",
    "Valhalla spd.": "valhalla_speed_limit",
    "Valhalla est.": "valhalla_estimate",
}
ABLATIONS = {
    "w/o Length & speed": "length_speed",
    "w/o Road properties": "road_properties",
    "w/o Link schedule": "link_schedule",
    "w/o Land use & POIs": "land_use_pois",
    "w/o Transit supply": "transit_supply",
    "w/o Stop position": "stop_position",
}
SUBSETS = {
    "train": "train",
    "val_temporal": "val-temp.",
    "val_spatial": "val-spat.",
    "test_temporal": "test-temp.",
    "test_spatial": "test-spat.",
}
METRICS = ["link/mase", "link/rmsse", "stop/mase", "stop/rmsse", "stop/f1"]
HEADER = f"{'':<22}{'Link MASE':>10}{'RMSSE':>8}{'Stop MASE':>11}{'RMSSE':>8}{'F1':>8}"


def load(results_dir: Path, run: str) -> Optional[Dict[str, float]]:
    path = results_dir / run / "metrics.json"
    if not path.exists():
        print(f"  (missing {path})")
        return None
    return json.loads(path.read_text())["aggregate"]


def row(label: str, values: List[Optional[float]], sign: bool = False) -> str:
    cells = [f"{v:{'+' if sign else ''}.4f}" if v is not None else "n/a" for v in values]
    widths = [10, 8, 11, 8, 8]
    return f"{label:<22}" + "".join(f"{c:>{w}}" for c, w in zip(cells, widths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    models = {label: load(args.results_dir, run) for label, run in MODELS.items()}
    baselines = load(args.results_dir, "baselines")

    print("\nTABLE III: GraphSAGE vs. GATv2 architectures and baselines (val-spatial)")
    print(HEADER)
    for label, agg in models.items():
        if agg is not None:
            print(row(label, [agg.get(f"val_spatial/{m}") for m in METRICS]))
    if baselines is not None:
        for label, name in BASELINES.items():
            print(row(label, [
                baselines.get(f"val_spatial/{name}/mase"),
                baselines.get(f"val_spatial/{name}/rmsse"),
                None, None, None,
            ]))

    print("\nTABLE IV: Temporal and spatial generalization")
    print(HEADER)
    for label in ("GATv2 + Linear", "GATv2 only", "Linear"):
        agg = models[label]
        if agg is None:
            continue
        print(label)
        for subset, subset_label in SUBSETS.items():
            print(row(f"  {subset_label}", [agg.get(f"{subset}/{m}") for m in METRICS]))

    base = models["GATv2 + Linear"]
    if base is not None:
        print("\nTABLE V: Grouped feature ablation (val-spatial, difference to GATv2 + Linear)")
        print(HEADER)
        print(row("Baseline", [base.get(f"val_spatial/{m}") for m in METRICS]))
        for label, run in ABLATIONS.items():
            agg = load(args.results_dir, f"ablation/{run}")
            if agg is not None:
                print(row(label, [
                    agg[f"val_spatial/{m}"] - base[f"val_spatial/{m}"] for m in METRICS
                ], sign=True))

        print("\nEmbedding diversity of GATv2 + Linear (Sec. VI-B)")
        for node in ("link", "stop"):
            print(f"  {node}: MAD {base[f'embedding/{node}/mad']:.4f}, "
                  f"MADGap {base[f'embedding/{node}/mad_gap']:.4f}")


if __name__ == "__main__":
    main()
