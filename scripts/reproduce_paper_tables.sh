#!/usr/bin/env bash
# Reproduce the experiments of the paper: Tables III-V, the embedding
# diversity (Sec. VI-B) and Fig. 4.
#
# Requires the graph artifacts and the preprocessed observations (see README).
#
# Usage:
#   scripts/reproduce_paper_tables.sh [all|models|baselines|ablation|figure|thresholds|tables|help]
#
#   models      Table III and IV: GNN variants and XGBoost (5 folds each)
#   baselines   Table III: rule-based baselines
#   ablation    Table V: GATv2 + Linear without one feature group at a time
#   figure      Fig. 4: PCA maps of the embeddings of The Hague
#   thresholds  stop classification thresholds (training/config.py)
#   tables      print Tables III-V from the results
#   all         models, baselines, ablation, figure and tables
#
# Environment variables:
#   DATA_DIR     directory with artifacts/ and preprocessed/ (default: data)
#   RESULTS_DIR  output directory (default: results)
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data}
RESULTS_DIR=${RESULTS_DIR:-results}
ARTIFACTS=$DATA_DIR/artifacts
DATA_ARGS=(
    --artifacts-dir "$ARTIFACTS"
    --travel-times-parquet "$DATA_DIR/preprocessed/travel_times.parquet"
    --dwell-times-parquet "$DATA_DIR/preprocessed/dwell_times.parquet"
)

train() {  # train <run> <train_embeddings arguments...>
    local run=$1
    shift
    mkdir -p "$RESULTS_DIR/$run"
    uv run python -m bus_embeddings.cli.train_embeddings "${DATA_ARGS[@]}" \
        --output-dir "$RESULTS_DIR/$run" "$@" 2>&1 | tee "$RESULTS_DIR/$run/train.log"
}

models() {
    train gat            --model gat
    train gat_only       --model gat --set global-skip-connection=false
    train linear         --model gat --set gnn-layers=0 --set learning-rate=0.0003
    train graphsage      --model graphsage
    train graphsage_only --model graphsage --set global-skip-connection=false
    train xgboost        --model xgboost
}

baselines() {
    mkdir -p "$RESULTS_DIR/baselines"
    uv run python -m bus_embeddings.cli.evaluate_baselines "${DATA_ARGS[@]}" \
        --output-dir "$RESULTS_DIR/baselines" 2>&1 | tee "$RESULTS_DIR/baselines/evaluate.log"
}

ablation() {
    local poi pois=()
    for poi in food-drink shopping education healthcare transport finance entertainment \
               sports-recreation tourism services worship other-amenity; do
        pois+=(--set "use-osm-poi-$poi=false")
    done
    train ablation/length_speed --model gat \
        --set use-osm-length=false --set use-osm-travel-time=false --set use-osm-speed=false
    train ablation/road_properties --model gat \
        --set use-osm-class=false --set use-osm-lanes=false \
        --set use-osm-traffic-lights=false --set use-osm-junctions=false
    train ablation/link_schedule --model gat \
        --set use-gtfs-link-pos=false --set use-gtfs-link-frequency=false
    train ablation/land_use_pois --model gat \
        "${pois[@]}" --set use-osm-landuse=false --set use-gisco-density=false
    train ablation/transit_supply --model gat \
        --set use-gtfs-trips=false --set use-gtfs-routes=false \
        --set use-gtfs-headway=false --set use-gtfs-poi-stops=false
    train ablation/stop_position --model gat \
        --set use-gtfs-stop-pos=false --set use-gtfs-terminus=false
}

figure() {
    uv run python scripts/plot_pca_embeddings.py --checkpoint "$RESULTS_DIR/gat/fold_4.pt" \
        --artifacts-dir "$ARTIFACTS" --output-dir "$RESULTS_DIR/figures"
}

thresholds() {
    for run in graphsage gat xgboost; do
        uv run python scripts/threshold_sweep.py --checkpoint-dir "$RESULTS_DIR/$run" "${DATA_ARGS[@]}"
    done
}

tables() {
    uv run python scripts/make_paper_tables.py --results-dir "$RESULTS_DIR"
}

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

case "${1:-all}" in
    all) models; baselines; ablation; figure; tables ;;
    models | baselines | ablation | figure | thresholds | tables) "$1" ;;
    help | -h | --help) usage ;;
    *) usage >&2; exit 1 ;;
esac
