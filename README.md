# Inductive Embeddings of Public Transport Networks

Code for the paper *Learning Inductive Embeddings of Public Transport Networks
Using Heterogeneous Graphs for Transfer to Unseen Municipalities* by Alexander
Horn, Rahul Mondal, Philip-Roman Adam, Patrick Cato and Stefanie Schmidtner.

The bus network of a municipality is modelled as a bipartite, heterogeneous
graph whose nodes are stops and links (connections between consecutive stops),
connected by four edge types (`stop –src_of→ link`, `stop –dst_of→ link`,
`link –to_src→ stop`, `link –to_dst→ stop`). Nodes carry static features from
GTFS schedules, OpenStreetMap (via Valhalla and OSMnx) and the Eurostat
population grid. A GraphSAGE or GATv2 encoder learns embeddings of the nodes
through three proxy tasks with separate prediction heads: travel time
prediction for links, and dwell time and stop probability prediction for stops.
As the node features use no historical observations, the model is evaluated on
municipalities that were not seen during training.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11. PyTorch is
installed with CUDA 12.8 support (Linux and Windows); training also runs on the
CPU, but slowly.

```bash
git clone https://github.com/urbanaithi/bus-embeddings.git
cd bus-embeddings
uv sync                 # add --extra wandb for Weights & Biases logging
```

## Data

The scripts expect the following files in `data/` (configurable with `DATA_DIR`):

| Path | Content |
|------|---------|
| `raw/travel_times/<GM code>.csv`, `raw/dwell_times/<GM code>.csv` | Travel and dwell times of the 18 Dutch municipalities of the benchmark dataset of Horn et al. (ITSC 2025, [doi:10.1109/ITSC60802.2025.11423733](https://doi.org/10.1109/ITSC60802.2025.11423733)) |
| `gtfs/NL-20250305.gtfs.zip` | GTFS feed of the Netherlands of 5 March 2025 |
| `census/ESTAT_Census_2021_V2.parquet` | Eurostat Census 2021 population grid (1 km², version 2.2) from [GISCO](https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/population-grids) |

Building the graphs additionally requires [GNU parallel](https://www.gnu.org/software/parallel/),
a [Valhalla](https://github.com/valhalla/valhalla) server with routing tiles of
the Netherlands (e.g. from the OpenStreetMap extract of
[Geofabrik](https://download.geofabrik.de/europe/netherlands.html)) and access
to the Overpass API, from which POIs and land use are downloaded.
Valhalla and Overpass responses are cached in `~/.cache/bus-embeddings/`.
Note that the OpenStreetMap data, and thus the node features, change over time.

```bash
VALHALLA_URL=http://localhost:8002 scripts/prepare_data.sh
```

builds the graph artifacts of every municipality (`data/artifacts/<GM code>/`)
and removes outliers from the observations (see
[`data/thresholds.py`](src/bus_embeddings/data/thresholds.py)), writing
`data/preprocessed/{travel,dwell}_times.parquet`.

## Reproducing the paper

```bash
scripts/reproduce_paper_tables.sh        # all experiments, results in results/
scripts/reproduce_paper_tables.sh tables # print Tables III-V again
```

The script trains the models of Tables III and IV (GATv2 + Linear, GATv2 only,
Linear, GraphSAGE + Linear, GraphSAGE only, XGBoost), evaluates the
rule-based baselines, runs the feature ablation of Table V, plots Fig. 4, and
prints the tables and the embedding diversity (MAD, MADGap) of Sec. VI-B. Run
`scripts/reproduce_paper_tables.sh help` for the individual steps.

## Training a model

```bash
uv run python -m bus_embeddings.cli.train_embeddings \
    --artifacts-dir data/artifacts \
    --travel-times-parquet data/preprocessed/travel_times.parquet \
    --dwell-times-parquet data/preprocessed/dwell_times.parquet \
    --model gat --output-dir results/gat
```

`--model` is `gat` (GATv2), `graphsage` or `xgboost`. The tuned hyperparameters
are the defaults in [`training/config.py`](src/bus_embeddings/training/config.py),
which also defines the region split and the node features. Any hyperparameter
can be overridden, e.g. `--set global-skip-connection=false` (GNN only) or
`--set gnn-layers=0` (Linear); the settings of all runs of the paper are in
`scripts/reproduce_paper_tables.sh`.

Each model is trained on five expanding temporal folds. The municipalities are
split into training regions (A), validation regions (B) and test regions (C);
the checkpoint with the best validation score (geometric mean of the link and
stop RMSSE on regions B) is evaluated on

| Subset | Regions | Period |
|--------|---------|--------|
| `train` | A | training |
| `val_temporal` | A | validation |
| `val_spatial` | B | validation |
| `test_temporal` | A | test |
| `test_spatial` | C | test |

with the MAE, RMSE, and the MASE and RMSSE relative to a weekly seasonal naive
forecast for links and stops, and the F1 score of the stop classification. The
decision threshold of the stop classifier was determined during development as
the F1-optimal threshold on the validation regions, averaged over the five
folds, and is fixed in `training/config.py`; `scripts/threshold_sweep.py`
computes it for trained models.
The run writes one checkpoint per fold (`fold_<i>.pt`) and `metrics.json` with
the metrics of every fold and their aggregate over the folds (geometric mean;
arithmetic mean for MADGap).

## Repository structure

```
src/bus_embeddings/
    cli/            command-line entry points
    graph/          graph construction and feature extraction (GTFS, OSM/Valhalla, Eurostat)
    data/           loading, filtering and sampling of the observations
    models/         GraphSAGE/GATv2 encoders with prediction heads, XGBoost baseline
    training/       configuration and evaluation metrics
    utils/          graph assembly and data splits
scripts/
    prepare_data.sh             build the graphs and preprocess the observations
    reproduce_paper_tables.sh   run all experiments of the paper
    make_paper_tables.py        print Tables III-V from the results
    plot_pca_embeddings.py      Fig. 4
    threshold_sweep.py          stop classification thresholds
```

## License

[MIT](LICENSE)
