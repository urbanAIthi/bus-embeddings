#!/usr/bin/env bash
# Build the graph artifacts of all municipalities and preprocess the
# observations (see README).
#
# Inputs (under $DATA_DIR, default: data):
#   raw/travel_times/<GM code>.csv, raw/dwell_times/<GM code>.csv
#   gtfs/NL-20250305.gtfs.zip               ($GTFS_ZIP)
#   census/ESTAT_Census_2021_V2.parquet     ($CENSUS_PATH)
# Outputs:
#   artifacts/<GM code>/{nodes_link,nodes_stop,edges}.csv
#   preprocessed/{travel_times,dwell_times}.parquet
#
# Requires GNU parallel and a Valhalla server with tiles of the Netherlands
# ($VALHALLA_URL, default: http://localhost:8002). POIs and land use are
# downloaded from the Overpass API with OSMnx. Valhalla and OSMnx responses
# are cached in ~/.cache/bus-embeddings/. $JOBS municipalities are built in
# parallel (default: 8).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data}
VALHALLA_URL=${VALHALLA_URL:-http://localhost:8002}
GTFS_ZIP=${GTFS_ZIP:-$DATA_DIR/gtfs/NL-20250305.gtfs.zip}
CENSUS_PATH=${CENSUS_PATH:-$DATA_DIR/census/ESTAT_Census_2021_V2.parquet}
JOBS=${JOBS:-8}

MUNICIPALITIES=(
    GM0047  # Veendam
    GM0059  # Achtkarspelen
    GM0281  # Tiel
    GM0312  # Bunnik
    GM0344  # Utrecht
    GM0503  # Delft
    GM0518  # Den Haag
    GM0546  # Leiden
    GM0590  # Papendrecht
    GM0599  # Rotterdam
    GM0629  # Wassenaar
    GM1681  # Borger-Odoorn
    GM1690  # De Wolden
    GM1731  # Midden-Drenthe
    GM1842  # Midden-Delfland
    GM1930  # Nissewaard
    GM1950  # Westerwolde
    GM1969  # Westerkwartier
)

build_one() {
    uv run python -m bus_embeddings.cli.build_graph \
        --travel-times "$DATA_DIR/raw/travel_times/$1.csv" \
        --output-dir "$DATA_DIR/artifacts/$1" \
        --valhalla-url "$VALHALLA_URL" \
        --gtfs-zip "$GTFS_ZIP" \
        --census-path "$CENSUS_PATH" \
        --osm-poi-buffer 500
}
export -f build_one
export DATA_DIR VALHALLA_URL GTFS_ZIP CENSUS_PATH

# Abort all jobs as soon as one fails; prefix the output with the GM code
parallel --halt now,fail=1 --line-buffer --tag -j "$JOBS" build_one ::: "${MUNICIPALITIES[@]}"

uv run python -m bus_embeddings.cli.preprocess_data \
    --travel-times-dir "$DATA_DIR/raw/travel_times" \
    --dwell-times-dir "$DATA_DIR/raw/dwell_times" \
    --artifacts-dir "$DATA_DIR/artifacts" \
    --output-dir "$DATA_DIR/preprocessed"
