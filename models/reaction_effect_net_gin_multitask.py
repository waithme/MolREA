from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool

from ..chemistry import mol_from_smiles
from ..data.graph_features import atom_features, bond_features


def mol_to_graph(smiles: str) -> Data:
    mol = mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return dummy_graph()

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.long)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_indices.extend([(begin, end), (end, begin)])
        edge_attrs.extend([features, features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def dummy_graph() -> Data:
    return Data(
        x=torch.zeros((1, 8), dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 4), dtype=torch.long),
    )


class GINGraphEncoder(nn.Module):
    """GIN graph encoder with mean pooling."""

    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(128, hidden_dim) for _ in range(node_dim)]
        )
        self.edge_embeddings = nn.ModuleList(
            [nn.Embedding(128, hidden_dim) for _ in range(edge_dim)]
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))

    def _encode_node_features(self, x: torch.Tensor) -> torch.Tensor:
        encoded = 0
        for feature_idx, embedding in enumerate(self.feature_embeddings):
            values = x[:, feature_idx].long().clamp(min=0, max=127)
            encoded = encoded + embedding(values)
        return encoded

    def _encode_edge_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.numel() == 0:
            return edge_attr.new_zeros((0, self.edge_embeddings[0].embedding_dim)).float()
        encoded = 0
        for feature_idx, embedding in enumerate(self.edge_embeddings):
            values = edge_attr[:, feature_idx].long().clamp(min=0, max=127)
            encoded = encoded + embedding(values)
        return encoded

    def forward(self, graph_batch: Batch) -> torch.Tensor:
        x = self._encode_node_features(graph_batch.x)
        edge_attr = self._encode_edge_features(graph_batch.edge_attr)
        for conv in self.convs:
            x = conv(x, graph_batch.edge_index, edge_attr)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, graph_batch.batch)


class FeatureAlignmentBlock(nn.Module):
    """Project FP/property vectors to 256-d and align them with GIN graph embeddings."""

    def __init__(self, fp_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fp_mlp = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_norm = nn.LayerNorm(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_values(self, val: torch.Tensor) -> torch.Tensor:
        value_feature = self.value_encoder(val.unsqueeze(-1))
        value_feature = value_feature.mean(dim=-2)
        return self.value_norm(value_feature)

    def forward(self, graph_emb: torch.Tensor, fp: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dropout(graph_emb + self.fp_mlp(fp) + self.encode_values(val)))

    def product_forward(self, graph_emb: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dropout(graph_emb + self.fp_mlp(fp)))


class TaskPredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.output(hidden), hidden


class AbsolutePredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_tasks: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim // 2, num_tasks)

    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.output(hidden), hidden


class ReactionEffectNetGINMultiTask(nn.Module):
    """Multi-task ReactionEffectNet with GIN mean pooling and per-metric heads."""

    def __init__(
        self,
        metric_names: Sequence[str],
        fp_dim: int = 2048,
        hidden_dim: int = 256,
        rxn_emb_dim: int = 256,
        num_gin_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        reactant_slots: int = 3,
    ) -> None:
        super().__init__()
        self.metric_names = tuple(metric_names)
        self.num_tasks = len(self.metric_names)
        self.reactant_slots = reactant_slots
        self.graph_encoder = GINGraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_gin_layers,
            dropout=dropout,
        )
        self.alignment = FeatureAlignmentBlock(fp_dim, hidden_dim, dropout)
        self.rxn_gate = nn.Sequential(
            nn.Linear(rxn_emb_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.reactant_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.prediction_heads = nn.ModuleList(
            [TaskPredictionHead(hidden_dim, dropout) for _ in range(self.num_tasks)]
        )
        self.absolute_prediction_head = AbsolutePredictionHead(hidden_dim, self.num_tasks, dropout)
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        self.register_buffer("metric_mean", torch.zeros(self.num_tasks))
        self.register_buffer("metric_std", torch.ones(self.num_tasks))

    def set_metric_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.metric_mean.copy_(mean.float())
        self.metric_std.copy_(std.float().clamp(min=1e-8))

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.metric_mean) / self.metric_std

    def encode(
        self,
        reactant_graph_batch: Batch,
        product_graph_batch: Batch,
        fp: torch.Tensor,
        val: torch.Tensor,
        c_fp: torch.Tensor,
        rxn_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = fp.size(0)
        reactant_graph_emb = self.graph_encoder(reactant_graph_batch)
        reactant_graph_emb = reactant_graph_emb.view(batch_size, self.reactant_slots, -1)
        product_graph_emb = self.graph_encoder(product_graph_batch)

        val = self.normalize_values(val)
        mask = mask.bool()
        slot_mask = mask.unsqueeze(-1).to(dtype=reactant_graph_emb.dtype)
        reactant_feature = self.alignment(reactant_graph_emb, fp, val)
        product_feature = self.alignment.product_forward(product_graph_emb, c_fp)

        gate = self.rxn_gate(rxn_emb).unsqueeze(1)
        reactant_feature = reactant_feature * gate * slot_mask

        attn_output = torch.zeros_like(reactant_feature)
        has_valid_slot = mask.any(dim=1)
        if bool(has_valid_slot.any()):
            valid_output, _ = self.reactant_attention(
                query=reactant_feature[has_valid_slot],
                key=reactant_feature[has_valid_slot],
                value=reactant_feature[has_valid_slot],
                key_padding_mask=~mask[has_valid_slot],
                need_weights=False,
            )
            attn_output[has_valid_slot] = valid_output
        reactant_feature = self.attn_norm(reactant_feature + self.attn_dropout(attn_output))
        reactant_feature = self.ffn_norm(reactant_feature + self.ffn(reactant_feature))
        reactant_feature = reactant_feature * slot_mask
        product_feature = product_feature.unsqueeze(1).expand(-1, self.reactant_slots, -1)
        return reactant_feature, product_feature

    def predict_delta_and_absolute(
        self,
        reactant_graph_batch: Batch,
        product_graph_batch: Batch,
        fp: torch.Tensor,
        val: torch.Tensor,
        c_fp: torch.Tensor,
        rxn_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_prediction, absolute_prediction, _, _ = self.predict_all(
            reactant_graph_batch=reactant_graph_batch,
            product_graph_batch=product_graph_batch,
            fp=fp,
            val=val,
            c_fp=c_fp,
            rxn_emb=rxn_emb,
            mask=mask,
        )
        return delta_prediction, absolute_prediction

    def _predict_delta_from_features(
        self,
        reactant_feature: torch.Tensor,
        product_feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_input = torch.cat([reactant_feature, product_feature], dim=-1)
        predictions = []
        delta_hidden = []
        for head in self.prediction_heads:
            pred, hidden = head.forward_with_hidden(delta_input)
            predictions.append(pred)
            delta_hidden.append(hidden)
        delta_prediction = torch.stack(predictions, dim=1)
        delta_hidden = torch.stack(delta_hidden, dim=1)
        task_slot_mask = mask.unsqueeze(1).unsqueeze(-1).to(dtype=delta_prediction.dtype)
        delta_prediction = delta_prediction * task_slot_mask
        delta_hidden = delta_hidden * task_slot_mask
        return delta_prediction, delta_hidden

    def predict_delta_only(
        self,
        reactant_graph_batch: Batch,
        product_graph_batch: Batch,
        fp: torch.Tensor,
        val: torch.Tensor,
        c_fp: torch.Tensor,
        rxn_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reactant_feature, product_feature = self.encode(
            reactant_graph_batch=reactant_graph_batch,
            product_graph_batch=product_graph_batch,
            fp=fp,
            val=val,
            c_fp=c_fp,
            rxn_emb=rxn_emb,
            mask=mask,
        )
        delta_prediction, delta_hidden = self._predict_delta_from_features(
            reactant_feature,
            product_feature,
            mask,
        )
        return reactant_feature, delta_prediction, delta_hidden

    def predict_all(
        self,
        reactant_graph_batch: Batch,
        product_graph_batch: Batch,
        fp: torch.Tensor,
        val: torch.Tensor,
        c_fp: torch.Tensor,
        rxn_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reactant_feature, product_feature = self.encode(
            reactant_graph_batch=reactant_graph_batch,
            product_graph_batch=product_graph_batch,
            fp=fp,
            val=val,
            c_fp=c_fp,
            rxn_emb=rxn_emb,
            mask=mask,
        )
        delta_prediction, delta_hidden = self._predict_delta_from_features(
            reactant_feature,
            product_feature,
            mask,
        )
        reactant_mask = mask.unsqueeze(-1).to(dtype=reactant_feature.dtype)
        reactant_summary = (reactant_feature * reactant_mask).sum(dim=1) / reactant_mask.sum(dim=1).clamp_min(1.0)
        product_summary = product_feature[:, 0, :]
        absolute_prediction, absolute_hidden = self.absolute_prediction_head.forward_with_hidden(
            torch.cat([reactant_summary, product_summary], dim=-1)
        )
        return delta_prediction, absolute_prediction, delta_hidden, absolute_hidden


def target_delta_normalized(c_val: torch.Tensor, val: torch.Tensor, metric_std: torch.Tensor) -> torch.Tensor:
    delta = c_val.unsqueeze(1) - val
    delta = delta.transpose(1, 2).unsqueeze(-1)
    return delta / metric_std.view(1, -1, 1, 1)


def load_backbone_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("backbone_state_dict", checkpoint.get("model_state_dict", checkpoint))
    current_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key.startswith("prediction_heads.") or key.startswith("absolute_prediction_head.") or key == "log_vars":
            skipped.append(key)
            continue
        if key not in current_state or current_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded backbone from {checkpoint_path}")
    print(
        f"Loaded keys: {len(filtered)}, Skipped keys: {len(skipped)}, "
        f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )
