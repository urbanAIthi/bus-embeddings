"""GNN embedding models for proxy-task training.

Learns graph embeddings by predicting travel / dwell times from static
features and time context with lightweight prediction heads.
"""
from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv

# Time context: sin_tod, cos_tod, dow_onehot[7], sin_month, cos_month, is_holiday
# Supports ablation via env vars ABLATE_NO_MONTH, ABLATE_NO_HOLIDAY
import os as _os
TIME_CONTEXT_DIM = 12
if _os.environ.get("ABLATE_NO_MONTH") == "1":
    TIME_CONTEXT_DIM -= 2
if _os.environ.get("ABLATE_NO_HOLIDAY") == "1":
    TIME_CONTEXT_DIM -= 1


class MLPPredictionHead(nn.Module):
    """MLP prediction head that optionally concatenates time context."""

    def __init__(
        self,
        embedding_dim: int,
        time_context_dim: int = TIME_CONTEXT_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        time_concat: bool = True,
    ) -> None:
        super().__init__()
        mlp_input_dim = embedding_dim + (time_context_dim if time_concat else 0)
        self.time_concat = time_concat
        self.predictor = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor, time_context: torch.Tensor) -> torch.Tensor:
        if self.time_concat:
            predictor_input = torch.cat([embedding, time_context], dim=-1)
        else:
            predictor_input = embedding
        return self.predictor(predictor_input).squeeze(-1)


class FiLMPredictionHead(nn.Module):
    """Prediction head using Feature-wise Linear Modulation (FiLM).

    Time context generates per-dimension scale and shift parameters that
    modulate the embedding, capturing how temporal context changes the
    interpretation of static features (e.g., traffic lights cause more
    delay at rush hour).

    When ``time_concat=True``, the predictor also receives the raw time
    context for additive time-dependent offsets independent of link features.
    """

    def __init__(
        self,
        embedding_dim: int,
        time_context_dim: int = TIME_CONTEXT_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        time_concat: bool = True,
        n_head_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_concat = time_concat

        self.film_generator = nn.Sequential(
            nn.Linear(time_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim * 2),
        )

        mlp_input_dim = embedding_dim + (time_context_dim if time_concat else 0)
        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for _ in range(n_head_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.predictor = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor, time_context: torch.Tensor) -> torch.Tensor:
        film_params = self.film_generator(time_context)
        scale, shift = film_params.chunk(2, dim=-1)
        scale = scale + 1.0  # centre at identity so initial modulation is a no-op
        modulated = scale * embedding + shift

        if self.time_concat:
            predictor_input = torch.cat([modulated, time_context], dim=-1)
        else:
            predictor_input = modulated
        return self.predictor(predictor_input).squeeze(-1)


class BaseEmbeddingGNN(nn.Module, ABC):
    """Abstract base for GNN embedding models with proxy task prediction heads.

    Three prediction heads: link regression (travel time), stop regression
    (dwell time), stop classification (stop probability).

    Subclasses build their own conv layers in ``__init__`` and call
    ``_forward_gnn`` for the shared message-passing loop.  They must set:
    ``self.convs``, ``self.layer_norms``, ``self.skip_projs``,
    ``self.global_skip_projs``.
    """

    _VALID_HEAD_TYPES = frozenset({"film", "mlp"})

    def __init__(
        self,
        link_features_in: int,
        stop_features_in: int,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 8,
        gnn_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 32,
        global_skip_connection: bool = False,
        head_type: str = "film",
        head_time_concat: bool = True,
        n_head_layers: int = 1,
        edge_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
        add_route_shared_edges: bool = False,
    ) -> None:
        super().__init__()

        if head_type not in self._VALID_HEAD_TYPES:
            raise ValueError(
                f"Invalid head_type='{head_type}'. "
                f"Expected one of {sorted(self._VALID_HEAD_TYPES)}"
            )

        self.link_features_in = link_features_in
        self.stop_features_in = stop_features_in
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_output_dim = gnn_output_dim
        self.gnn_layers = gnn_layers
        self.dropout = dropout
        self.head_hidden_dim = head_hidden_dim
        self.global_skip_connection = global_skip_connection
        self.head_type = head_type
        self.head_time_concat = head_time_concat
        self.add_route_shared_edges = add_route_shared_edges

        self.dropout_layer = nn.Dropout(dropout)
        self.edge_dropout = edge_dropout
        self.feature_noise_std = feature_noise_std

        # With global skip AND gnn_layers > 0, raw features are concatenated -> 2x embedding dim.
        # With only one path (GNN-only or skip-only) or neither, dim = gnn_output_dim.
        if global_skip_connection and gnn_layers > 0:
            effective_dim = gnn_output_dim * 2
        else:
            effective_dim = gnn_output_dim

        # LayerNorm removes region-specific distribution shifts for spatial transfer
        self.embedding_norm = nn.LayerNorm(effective_dim)

        # Prediction heads
        head_cls = FiLMPredictionHead if head_type == "film" else MLPPredictionHead
        head_kwargs = dict(
            embedding_dim=effective_dim,
            time_context_dim=TIME_CONTEXT_DIM,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            time_concat=head_time_concat,
        )
        if head_type == "film":
            head_kwargs["n_head_layers"] = n_head_layers
        self.link_head = head_cls(**head_kwargs)        # travel time regression
        self.stop_head = head_cls(**head_kwargs)        # dwell time regression
        self.stop_classifier = head_cls(**head_kwargs)  # stop probability

    def _forward_gnn(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        activation: Callable,
    ) -> Dict[str, torch.Tensor]:
        """Message-passing loop shared by GraphSAGE and GAT subclasses.

        Per layer: HeteroConv → LayerNorm → residual skip → activation+dropout
        (last layer skips the activation). With ``gnn_layers=0`` returns the
        global-skip projection or zero embeddings (time-context only).
        """
        if len(self.convs) == 0:
            if self.global_skip_projs is not None:
                return {k: self.global_skip_projs[k](v) for k, v in x_dict.items()}
            return {
                k: torch.zeros(v.shape[0], self.gnn_output_dim, device=v.device, dtype=v.dtype)
                for k, v in x_dict.items()
            }

        # Feature noise during training improves transfer to unseen regions.
        if self.training and self.feature_noise_std > 0:
            h = {k: v + torch.randn_like(v) * self.feature_noise_std for k, v in x_dict.items()}
        else:
            h = x_dict

        for i, conv in enumerate(self.convs):
            residual = h

            # DropEdge regularization during training.
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

            # Skip activation on last layer to preserve linear output.
            if i < len(self.convs) - 1:
                h = {k: self.dropout_layer(activation(v)) for k, v in h.items()}

        if self.global_skip_projs is not None:
            h = {
                k: torch.cat([v, self.global_skip_projs[k](x_dict[k])], dim=-1)
                for k, v in h.items()
            }

        return h

    @abstractmethod
    def _compute_hetero_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute raw embeddings before LayerNorm."""

    def compute_embeddings(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute LayerNorm'd embeddings for all nodes in the graph."""
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
        if self.add_route_shared_edges:
            edge_index_dict[("link", "near", "link")] = batch["link", "near", "link"].edge_index
            edge_index_dict[("stop", "near", "stop")] = batch["stop", "near", "stop"].edge_index
        embeddings = self._compute_hetero_embeddings(x_dict, edge_index_dict)
        return {k: self.embedding_norm(v) for k, v in embeddings.items()}

    def forward(
        self,
        batch: HeteroData,
        link_node_indices: torch.Tensor,
        link_time_context: torch.Tensor,
        stop_node_indices: torch.Tensor,
        stop_time_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute embeddings once per batch, then predict per sample.

        Returns ``(link_preds, stop_preds, stop_logits, embeddings)``.
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
    """GraphSAGE embedding model using HeteroConv with SAGEConv.

    Each conv layer wraps four SAGEConv instances for the bipartite edge
    types between stop and link nodes::

        stop --(src_of)--> link : SAGEConv((dim_stop_feat, dim_link_feat), dim_out)
        stop --(dst_of)--> link : SAGEConv((dim_stop_feat, dim_link_feat), dim_out)
        link --(to_src)--> stop : SAGEConv((dim_link_feat, dim_stop_feat), dim_out)
        link --(to_dst)--> stop : SAGEConv((dim_link_feat, dim_stop_feat), dim_out)

    Layer stack:  input -> [hidden]x(L-1) -> output.
    Activation: ReLU between layers.
    """

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
        head_time_concat: bool = True,
        n_head_layers: int = 1,
        edge_dropout: float = 0.0,
        sage_aggr: str = "mean",
        feature_noise_std: float = 0.0,
        add_route_shared_edges: bool = False,
    ) -> None:
        super().__init__(
            link_features_in=link_features_in,
            stop_features_in=stop_features_in,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_layers=gnn_layers,
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
            global_skip_connection=global_skip_connection,
            head_type=head_type,
            head_time_concat=head_time_concat,
            n_head_layers=n_head_layers,
            edge_dropout=edge_dropout,
            feature_noise_std=feature_noise_std,
            add_route_shared_edges=add_route_shared_edges,
        )

        # Per-layer dimensions: (link_in, stop_in, out)
        L, S, H, O = link_features_in, stop_features_in, gnn_hidden_dim, gnn_output_dim

        # Build conv stack: each layer is a HeteroConv wrapping 4 SAGEConv
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleDict()
        self.skip_projs = nn.ModuleDict()

        if gnn_layers > 0:
            if gnn_layers == 1:
                layer_dims = [(L, S, O)]
            else:
                layer_dims = [(L, S, H)] + [(H, H, H)] * (gnn_layers - 2) + [(H, H, O)]

            for i, (l_in, s_in, out) in enumerate(layer_dims):
                conv_dict = {
                    ("stop", "src_of", "link"): SAGEConv((s_in, l_in), out, aggr=sage_aggr),
                    ("stop", "dst_of", "link"): SAGEConv((s_in, l_in), out, aggr=sage_aggr),
                    ("link", "to_src", "stop"): SAGEConv((l_in, s_in), out, aggr=sage_aggr),
                    ("link", "to_dst", "stop"): SAGEConv((l_in, s_in), out, aggr=sage_aggr),
                }
                if add_route_shared_edges:
                    conv_dict[("link", "near", "link")] = SAGEConv((l_in, l_in), out, aggr=sage_aggr)
                    conv_dict[("stop", "near", "stop")] = SAGEConv((s_in, s_in), out, aggr=sage_aggr)
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))

                self.layer_norms[str(i)] = nn.ModuleDict({
                    "link": nn.LayerNorm(out),
                    "stop": nn.LayerNorm(out),
                })

                # Residual projection when input and output dims differ
                if skip_connections and (l_in != out or s_in != out):
                    self.skip_projs[str(i)] = nn.ModuleDict({
                        "link": nn.Linear(l_in, out, bias=False),
                        "stop": nn.Linear(s_in, out, bias=False),
                    })

        # Global skip: project raw input features -> gnn_output_dim, concatenated
        self.global_skip_projs: nn.ModuleDict | None = None
        if global_skip_connection:
            self.global_skip_projs = nn.ModuleDict({
                "link": nn.Linear(link_features_in, gnn_output_dim, bias=False),
                "stop": nn.Linear(stop_features_in, gnn_output_dim, bias=False),
            })

    def _compute_hetero_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self._forward_gnn(x_dict, edge_index_dict, activation=torch.relu)


class GATEmbedding(BaseEmbeddingGNN):
    """GAT embedding model using HeteroConv with GATv2Conv.

    Each conv layer wraps four GATv2Conv instances.  Interior layers use
    multi-head attention (``num_heads``, concatenated outputs); the final
    layer uses a single head (no concatenation).

    Layer stack:  input -> [hidden]x(L-1) -> output.
    Activation: LeakyReLU between layers.
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
        head_time_concat: bool = True,
        n_head_layers: int = 1,
        edge_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
        add_route_shared_edges: bool = False,
    ) -> None:
        super().__init__(
            link_features_in=link_features_in,
            stop_features_in=stop_features_in,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_layers=gnn_layers,
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
            global_skip_connection=global_skip_connection,
            head_type=head_type,
            head_time_concat=head_time_concat,
            n_head_layers=n_head_layers,
            edge_dropout=edge_dropout,
            feature_noise_std=feature_noise_std,
            add_route_shared_edges=add_route_shared_edges,
        )
        self.num_heads = num_heads
        # Separate attention dropout: defaults to feature dropout if not set
        self.attn_dropout = attn_dropout if attn_dropout is not None else dropout

        if num_heads > 1 and gnn_hidden_dim % num_heads != 0:
            raise ValueError(
                f"gnn_hidden_dim ({gnn_hidden_dim}) must be divisible by "
                f"num_heads ({num_heads}) for GAT multi-head attention."
            )

        # Per-layer dimensions: (link_in, stop_in, out)
        L, S, H, O = link_features_in, stop_features_in, gnn_hidden_dim, gnn_output_dim

        # Build conv stack: each layer is a HeteroConv wrapping 4 GATv2Conv
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleDict()
        self.skip_projs = nn.ModuleDict()

        if gnn_layers > 0:
            if gnn_layers == 1:
                layer_dims = [(L, S, O)]
            else:
                layer_dims = [(L, S, H)] + [(H, H, H)] * (gnn_layers - 2) + [(H, H, O)]

            for i, (l_in, s_in, out) in enumerate(layer_dims):
                # Multi-head on interior layers, single head on last layer
                is_last = (i == len(layer_dims) - 1)
                heads = 1 if is_last else num_heads
                concat = heads > 1
                per_head = out // heads if concat else out

                conv_dict = {
                    ("stop", "src_of", "link"): GATv2Conv(
                        (s_in, l_in), per_head,
                        heads=heads, concat=concat, dropout=self.attn_dropout, add_self_loops=False,
                    ),
                    ("stop", "dst_of", "link"): GATv2Conv(
                        (s_in, l_in), per_head,
                        heads=heads, concat=concat, dropout=self.attn_dropout, add_self_loops=False,
                    ),
                    ("link", "to_src", "stop"): GATv2Conv(
                        (l_in, s_in), per_head,
                        heads=heads, concat=concat, dropout=self.attn_dropout, add_self_loops=False,
                    ),
                    ("link", "to_dst", "stop"): GATv2Conv(
                        (l_in, s_in), per_head,
                        heads=heads, concat=concat, dropout=self.attn_dropout, add_self_loops=False,
                    ),
                }
                if add_route_shared_edges:
                    conv_dict[("link", "near", "link")] = GATv2Conv(
                        (l_in, l_in), per_head,
                        heads=heads, concat=concat, dropout=self.attn_dropout, add_self_loops=False,
                    )
                    conv_dict[("stop", "near", "stop")] = GATv2Conv(
                        (s_in, s_in), per_head,
                        heads=heads, concat=concat, dropout=self.attn_dropout, add_self_loops=False,
                    )
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))

                self.layer_norms[str(i)] = nn.ModuleDict({
                    "link": nn.LayerNorm(out),
                    "stop": nn.LayerNorm(out),
                })

                if skip_connections and (l_in != out or s_in != out):
                    self.skip_projs[str(i)] = nn.ModuleDict({
                        "link": nn.Linear(l_in, out, bias=False),
                        "stop": nn.Linear(s_in, out, bias=False),
                    })

        # Global skip: project raw input features -> gnn_output_dim, concatenated
        self.global_skip_projs: nn.ModuleDict | None = None
        if global_skip_connection:
            self.global_skip_projs = nn.ModuleDict({
                "link": nn.Linear(link_features_in, gnn_output_dim, bias=False),
                "stop": nn.Linear(stop_features_in, gnn_output_dim, bias=False),
            })

    def _compute_hetero_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self._forward_gnn(x_dict, edge_index_dict, activation=F.leaky_relu)
