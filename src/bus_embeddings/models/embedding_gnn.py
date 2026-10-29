"""GNN encoders with prediction heads for the proxy tasks.

The encoder embeds the link and stop nodes of the bipartite graph with
heterogeneous GraphSAGE or GATv2 convolutions (separate weights per edge
type). Three prediction heads map a node embedding and the time context of
an observation to a travel time (links), a dwell time and a stop
probability (stops).
"""
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

from bus_embeddings.data.proxy_task_dataset import TIME_CONTEXT_DIM

EdgeType = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------


class MLPPredictionHead(nn.Module):
    """MLP with one hidden layer on the embedding concatenated with the time context."""

    def __init__(
        self,
        embedding_dim: int,
        time_context_dim: int = TIME_CONTEXT_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + time_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor, time_context: torch.Tensor) -> torch.Tensor:
        """``[batch, dim_embed]``, ``[batch, dim_time]`` -> ``[batch]``."""
        return self.predictor(torch.cat([embedding, time_context], dim=-1)).squeeze(-1)


class FiLMPredictionHead(nn.Module):
    """Prediction head with feature-wise linear modulation (FiLM).

    The time context generates a per-dimension scale and shift that modulate
    the embedding; an MLP with ``n_head_layers`` hidden layers predicts from
    the modulated embedding concatenated with the time context.
    """

    def __init__(
        self,
        embedding_dim: int,
        time_context_dim: int = TIME_CONTEXT_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        n_head_layers: int = 1,
    ) -> None:
        super().__init__()
        # time_context [B, T] -> (scale, shift) [B, 2*D]
        self.film_generator = nn.Sequential(
            nn.Linear(time_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim * 2),
        )

        layers: list[nn.Module] = []
        in_dim = embedding_dim + time_context_dim
        for _ in range(n_head_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.predictor = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor, time_context: torch.Tensor) -> torch.Tensor:
        """``[batch, dim_embed]``, ``[batch, dim_time]`` -> ``[batch]``."""
        scale, shift = self.film_generator(time_context).chunk(2, dim=-1)
        # Centre the scale at 1 so that the initial modulation is the identity
        modulated = (scale + 1.0) * embedding + shift
        return self.predictor(torch.cat([modulated, time_context], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class BaseEmbeddingGNN(nn.Module):
    """Heterogeneous GNN encoder with link regression, stop regression and
    stop classification heads.

    Each layer applies a ``HeteroConv`` over the four edge types, LayerNorm per
    node type, a residual connection (``skip_connections``) and, except for
    the last layer, the activation and dropout. With
    ``global_skip_connection`` a linear projection of the input features is
    concatenated to the GNN output ("GNN + Linear"); with ``gnn_layers=0``
    the embedding is that projection alone ("Linear").

    Subclasses define ``MODEL_TYPE``, ``activation`` and ``_conv`` and call
    ``_build_layers`` at the end of their constructor.
    """

    MODEL_TYPE: str
    HEAD_TYPES = ("film", "mlp")

    activation: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        gnn_output_dim: int = 8,
        gnn_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 32,
        global_skip_connection: bool = False,
        head_type: str = "film",
        n_head_layers: int = 1,
        edge_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if head_type not in self.HEAD_TYPES:
            raise ValueError(f"Invalid head_type={head_type!r}, expected one of {self.HEAD_TYPES}")
        if gnn_layers == 0 and not global_skip_connection:
            raise ValueError("gnn_layers=0 requires global_skip_connection=True")

        self.dropout_layer = nn.Dropout(dropout)
        self.edge_dropout = edge_dropout
        self.feature_noise_std = feature_noise_std

        # GNN + Linear concatenates two embeddings of size gnn_output_dim
        if global_skip_connection and gnn_layers > 0:
            embedding_dim = gnn_output_dim * 2
        else:
            embedding_dim = gnn_output_dim
        self.embedding_norm = nn.LayerNorm(embedding_dim)

        head_kwargs: Dict[str, Any] = dict(
            embedding_dim=embedding_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        if head_type == "film":
            head_cls: type[nn.Module] = FiLMPredictionHead
            head_kwargs["n_head_layers"] = n_head_layers
        else:
            head_cls = MLPPredictionHead
        self.link_head = head_cls(**head_kwargs)        # travel time regression
        self.stop_head = head_cls(**head_kwargs)        # dwell time regression
        self.stop_classifier = head_cls(**head_kwargs)  # stop probability

    def _conv(self, in_channels: Tuple[int, int], out_channels: int, is_last: bool) -> nn.Module:
        """Convolution for one edge type from (source, target) feature sizes."""
        raise NotImplementedError

    def _build_layers(
        self,
        link_features_in: int,
        stop_features_in: int,
        gnn_hidden_dim: int,
        gnn_output_dim: int,
        gnn_layers: int,
        skip_connections: bool,
        global_skip_connection: bool,
    ) -> None:
        L, S, H, O = link_features_in, stop_features_in, gnn_hidden_dim, gnn_output_dim

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleDict()
        self.skip_projs = nn.ModuleDict()

        if gnn_layers > 0:
            if gnn_layers == 1:
                layer_dims = [(L, S, O)]
            else:
                layer_dims = [(L, S, H)] + [(H, H, H)] * (gnn_layers - 2) + [(H, H, O)]

            for i, (l_in, s_in, out) in enumerate(layer_dims):
                is_last = i == len(layer_dims) - 1
                conv_dict = {
                    ("stop", "src_of", "link"): self._conv((s_in, l_in), out, is_last),
                    ("stop", "dst_of", "link"): self._conv((s_in, l_in), out, is_last),
                    ("link", "to_src", "stop"): self._conv((l_in, s_in), out, is_last),
                    ("link", "to_dst", "stop"): self._conv((l_in, s_in), out, is_last),
                }
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))

                self.layer_norms[str(i)] = nn.ModuleDict({
                    "link": nn.LayerNorm(out),
                    "stop": nn.LayerNorm(out),
                })

                # Project the residual when input and output dims differ
                if skip_connections and (l_in != out or s_in != out):
                    self.skip_projs[str(i)] = nn.ModuleDict({
                        "link": nn.Linear(l_in, out, bias=False),
                        "stop": nn.Linear(s_in, out, bias=False),
                    })

        # Global skip: linear projection of the input features
        self.global_skip_projs: nn.ModuleDict | None = None
        if global_skip_connection:
            self.global_skip_projs = nn.ModuleDict({
                "link": nn.Linear(link_features_in, gnn_output_dim, bias=False),
                "stop": nn.Linear(stop_features_in, gnn_output_dim, bias=False),
            })

    def _forward_gnn(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Message passing: ``{type: [num_nodes, dim_in]}`` -> ``{type: [num_nodes, dim_embed]}``."""
        if len(self.convs) == 0:
            assert self.global_skip_projs is not None
            return {k: self.global_skip_projs[k](v) for k, v in x_dict.items()}

        # Gaussian noise on the input features during training
        if self.training and self.feature_noise_std > 0:
            h = {k: v + torch.randn_like(v) * self.feature_noise_std for k, v in x_dict.items()}
        else:
            h = x_dict

        for i, conv in enumerate(self.convs):
            residual = h

            # DropEdge: randomly remove edges during training
            if self.training and self.edge_dropout > 0:
                ei = {}
                for k, v in edge_index_dict.items():
                    mask = torch.rand(v.size(1), device=v.device) > self.edge_dropout
                    ei[k] = v[:, mask]
                h = conv(h, ei)
            else:
                h = conv(h, edge_index_dict)

            norms = self.layer_norms[str(i)]
            h = {k: norms[k](v) for k, v in h.items()}

            if self.skip_projs:
                key = str(i)
                if key in self.skip_projs:
                    proj = self.skip_projs[key]
                    h = {k: v + proj[k](residual[k]) for k, v in h.items()}
                else:
                    h = {k: v + residual[k] for k, v in h.items()}

            # No activation after the last layer
            if i < len(self.convs) - 1:
                h = {k: self.dropout_layer(self.activation(v)) for k, v in h.items()}

        if self.global_skip_projs is not None:
            h = {
                k: torch.cat([v, self.global_skip_projs[k](x_dict[k])], dim=-1)
                for k, v in h.items()
            }
        return h

    def compute_embeddings(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        """LayerNorm'd embeddings of all link and stop nodes of the graph."""
        x_dict = {
            "link": batch["link"].x,
            "stop": batch["stop"].x,
        }
        edge_index_dict = {
            ("stop", "src_of", "link"): batch["stop", "src_of", "link"].edge_index,
            ("stop", "dst_of", "link"): batch["stop", "dst_of", "link"].edge_index,
            ("link", "to_src", "stop"): batch["link", "to_src", "stop"].edge_index,
            ("link", "to_dst", "stop"): batch["link", "to_dst", "stop"].edge_index,
        }
        embeddings = self._forward_gnn(x_dict, edge_index_dict)
        return {k: self.embedding_norm(v) for k, v in embeddings.items()}

    def forward(
        self,
        batch: HeteroData,
        link_node_indices: torch.Tensor,
        link_time_context: torch.Tensor,
        stop_node_indices: torch.Tensor,
        stop_time_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Embed the whole graph and predict for the given observations.

        Args:
            batch: Graph with node features and edge indices.
            link_node_indices: Link node of each link observation, ``[batch_link]``.
            link_time_context: Time context of each link observation, ``[batch_link, dim_time]``.
            stop_node_indices: Stop node of each stop observation, ``[batch_stop]``.
            stop_time_context: Time context of each stop observation, ``[batch_stop, dim_time]``.

        Returns:
            Travel time predictions ``[batch_link]``, dwell time predictions
            ``[batch_stop]``, stop logits ``[batch_stop]`` and the embeddings
            of all nodes.
        """
        embeddings = self.compute_embeddings(batch)
        device = embeddings["link"].device

        if link_node_indices.numel() > 0:
            link_emb = embeddings["link"][link_node_indices]
            link_preds = self.link_head(link_emb, link_time_context)
        else:
            link_preds = torch.empty(0, device=device)

        if stop_node_indices.numel() > 0:
            stop_emb = embeddings["stop"][stop_node_indices]
            stop_preds = self.stop_head(stop_emb, stop_time_context)
            stop_logits = self.stop_classifier(stop_emb, stop_time_context)
        else:
            stop_preds = torch.empty(0, device=device)
            stop_logits = torch.empty(0, device=device)

        return link_preds, stop_preds, stop_logits, embeddings


class GraphSAGEEmbedding(BaseEmbeddingGNN):
    """GraphSAGE encoder (mean aggregation, ReLU)."""

    MODEL_TYPE = "graphsage"

    def __init__(
        self,
        link_features_in: int,
        stop_features_in: int,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 8,
        gnn_layers: int = 2,
        skip_connections: bool = True,
        global_skip_connection: bool = False,
        dropout: float = 0.1,
        head_hidden_dim: int = 32,
        head_type: str = "film",
        n_head_layers: int = 1,
        edge_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
    ) -> None:
        super().__init__(
            gnn_output_dim=gnn_output_dim,
            gnn_layers=gnn_layers,
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
            global_skip_connection=global_skip_connection,
            head_type=head_type,
            n_head_layers=n_head_layers,
            edge_dropout=edge_dropout,
            feature_noise_std=feature_noise_std,
        )
        self.activation = torch.relu
        self._build_layers(
            link_features_in, stop_features_in, gnn_hidden_dim, gnn_output_dim,
            gnn_layers, skip_connections, global_skip_connection,
        )

    def _conv(self, in_channels: Tuple[int, int], out_channels: int, is_last: bool) -> nn.Module:
        return SAGEConv(in_channels, out_channels, aggr="mean")


class GATEmbedding(BaseEmbeddingGNN):
    """GATv2 encoder (LeakyReLU).

    Interior layers use ``num_heads`` attention heads with concatenated
    outputs; the last layer uses a single head.
    """

    MODEL_TYPE = "gat"

    def __init__(
        self,
        link_features_in: int,
        stop_features_in: int,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 8,
        gnn_layers: int = 2,
        skip_connections: bool = True,
        global_skip_connection: bool = False,
        dropout: float = 0.1,
        attn_dropout: float | None = None,
        num_heads: int = 4,
        head_hidden_dim: int = 32,
        head_type: str = "film",
        n_head_layers: int = 1,
        edge_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
    ) -> None:
        super().__init__(
            gnn_output_dim=gnn_output_dim,
            gnn_layers=gnn_layers,
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
            global_skip_connection=global_skip_connection,
            head_type=head_type,
            n_head_layers=n_head_layers,
            edge_dropout=edge_dropout,
            feature_noise_std=feature_noise_std,
        )
        if num_heads > 1 and gnn_hidden_dim % num_heads != 0:
            raise ValueError(
                f"gnn_hidden_dim ({gnn_hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.activation = F.leaky_relu
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout if attn_dropout is not None else dropout
        self._build_layers(
            link_features_in, stop_features_in, gnn_hidden_dim, gnn_output_dim,
            gnn_layers, skip_connections, global_skip_connection,
        )

    def _conv(self, in_channels: Tuple[int, int], out_channels: int, is_last: bool) -> nn.Module:
        heads = 1 if is_last else self.num_heads
        concat = heads > 1
        return GATv2Conv(
            in_channels,
            out_channels // heads if concat else out_channels,
            heads=heads,
            concat=concat,
            dropout=self.attn_dropout,
            add_self_loops=False,
        )


def build_model(
    model_type: str,
    hyperparams: Dict[str, Any],
    link_features_in: int,
    stop_features_in: int,
) -> BaseEmbeddingGNN:
    """Instantiate a GNN embedding model from its hyperparameters."""
    common: Dict[str, Any] = dict(
        link_features_in=link_features_in,
        stop_features_in=stop_features_in,
        gnn_hidden_dim=hyperparams["gnn-hidden-dim"],
        gnn_output_dim=hyperparams["gnn-output-dim"],
        gnn_layers=hyperparams["gnn-layers"],
        skip_connections=hyperparams["skip-connections"],
        global_skip_connection=hyperparams["global-skip-connection"],
        dropout=hyperparams["dropout"],
        head_hidden_dim=hyperparams["head-hidden-dim"],
        head_type=hyperparams["head-type"],
        n_head_layers=hyperparams["n-head-layers"],
        edge_dropout=hyperparams["edge-dropout"],
        feature_noise_std=hyperparams["feature-noise-std"],
    )
    if model_type == "graphsage":
        return GraphSAGEEmbedding(**common)
    if model_type == "gat":
        return GATEmbedding(
            **common,
            num_heads=hyperparams["num-heads"],
            attn_dropout=hyperparams["attn-dropout"],
        )
    raise ValueError(f"Unknown GNN model type: {model_type!r}")
