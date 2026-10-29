#!/usr/bin/env python3
"""Plot the first principal component of the embeddings of The Hague (Fig. 4).

Computes the link and stop embeddings of the regions that contain The Hague
(GM0518) with a trained model, reduces the embeddings of The Hague to one
dimension with PCA, and plots them at their locations (EPSG:28992); links
are drawn at the midpoint between their stops.

Usage::

    python scripts/plot_pca_embeddings.py \\
        --checkpoint results/gat/fold_4.pt \\
        --artifacts-dir data/artifacts \\
        --output-dir results/figures
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

from bus_embeddings.models.checkpoint import load_checkpoint
from bus_embeddings.utils.hetero_graph_utils import build_hetero_graph
from bus_embeddings.utils.splits import region_artifact_paths

DEN_HAAG = "GM0518"

# Styling for a two-column paper; embed fonts in the PDF
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "TeX Gyre Termes", "Times", "DejaVu Serif"]
_FIG_SIZE = (3.8, 3.5)
_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
}


def wgs84_to_rd(lon: np.ndarray, lat: np.ndarray):
    """WGS84 lon/lat to Amersfoort / RD New (EPSG:28992)."""
    points = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs("EPSG:28992")
    return points.x.to_numpy(), points.y.to_numpy()


def den_haag_embeddings(model, checkpoint, artifacts_dir: Path, device: torch.device):
    """Embeddings and RD coordinates of the link and stop nodes of The Hague.

    The graph contains all regions of the split that includes The Hague, as
    during evaluation; node features are normalized with the training
    statistics of the checkpoint.
    """
    regions = next(
        checkpoint[key] for key in ("val_regions", "test_regions", "train_regions")
        if DEN_HAAG in checkpoint[key]
    )
    graph, link_nodes, stop_nodes, _ = build_hetero_graph(
        *region_artifact_paths(regions, artifacts_dir),
        feature_config=checkpoint["hyperparams"],
        feature_norm_stats=checkpoint["feature_norm_stats"],
    )
    with torch.no_grad():
        embeddings = model.compute_embeddings(graph.to(device))

    # Nodes of each region are contiguous, in the order of `regions`
    sizes = {
        node_type: [len(pd.read_csv(path, usecols=["id"])) for path in paths]
        for node_type, paths in zip(("link", "stop"), region_artifact_paths(regions, artifacts_dir)[:2])
    }
    i = regions.index(DEN_HAAG)
    result = {}
    for node_type, nodes in (("link", link_nodes), ("stop", stop_nodes)):
        start = sum(sizes[node_type][:i])
        end = start + sizes[node_type][i]
        result[f"{node_type}_emb"] = embeddings[node_type][start:end].cpu().numpy()
        result[f"{node_type}_nodes"] = nodes.iloc[start:end]

    links = result["link_nodes"]
    result["link_x"], result["link_y"] = wgs84_to_rd(
        (links["from_lon"].values + links["to_lon"].values) / 2,
        (links["from_lat"].values + links["to_lat"].values) / 2,
    )
    stops = result["stop_nodes"]
    result["stop_x"], result["stop_y"] = wgs84_to_rd(stops["lon"].values, stops["lat"].values)
    return result


def plot_pc1(x, y, pc1, explained_variance, out_path: Path, xlim, ylim) -> None:
    """Scatter plot of nodes at their RD coordinates coloured by PC1."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        sc = ax.scatter(
            x, y, c=pc1, cmap="coolwarm", s=3, alpha=0.6,
            norm=mcolors.CenteredNorm(), rasterized=True,
        )
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(f"PC1 ({explained_variance:.1%} var)")
        cbar.locator = matplotlib.ticker.MaxNLocator(integer=True)
        cbar.update_ticks()
        ax.set_xlabel("RD x (m)")
        ax.set_ylabel("RD y (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA maps of the embeddings of The Hague (Fig. 4).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint of a GNN (fold_<i>.pt).")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Graph artifacts directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    data = den_haag_embeddings(model, checkpoint, args.artifacts_dir, device)

    # Shared axis limits for both plots
    all_x = np.concatenate([data["link_x"], data["stop_x"]])
    all_y = np.concatenate([data["link_y"], data["stop_y"]])
    pad_x = (all_x.max() - all_x.min()) * 0.02
    pad_y = (all_y.max() - all_y.min()) * 0.02
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

    for node_type in ("link", "stop"):
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(data[f"{node_type}_emb"]).ravel()
        explained_variance = pca.explained_variance_ratio_[0]
        print(f"{node_type}: {len(pc1)} nodes, PC1 explains {explained_variance:.1%} of the variance")
        plot_pc1(
            data[f"{node_type}_x"], data[f"{node_type}_y"], pc1, explained_variance,
            args.output_dir / f"pca_{checkpoint['model_type']}_denhaag_{node_type}.pdf",
            xlim, ylim,
        )


if __name__ == "__main__":
    main()
