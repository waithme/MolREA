import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .data.dataset_specs import DatasetSpec, resolve_path
from .data.reaction_dataset import ReactionEffectDataset
from .data.splits import make_subsets
from .models.reaction_effect_net_gin_multitask import GINGraphEncoder, ReactionEffectNetGINMultiTask


@dataclass
class FusionConfig:
    reaction_checkpoint_path: str = "pretrain_backbone/best.pt"
    output_dir: str = "results/fusion"
    model_init_seed: int = 0
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 100
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 10
    reactant_slots: int = 3
    fp_dim: int = 2048
    hidden_dim: int = 256
    num_gin_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.0
    freeze_reaction_model: bool = True
    use_absolute_branch: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_model_with_seed(builder, seed: int):
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return builder()
    finally:
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def collate_fusion(batch):
    reactant_graphs = []
    product_graphs = []
    for item in batch:
        reactant_graphs.extend(item["reactant_graphs"])
        product_graphs.append(item["product_graph"])
    return {
        "reactant_graph_batch": Batch.from_data_list(reactant_graphs),
        "product_graph_batch": Batch.from_data_list(product_graphs),
        "fp": torch.stack([item["fp"] for item in batch], dim=0),
        "val": torch.stack([item["val"] for item in batch], dim=0),
        "c_fp": torch.stack([item["c_fp"] for item in batch], dim=0),
        "c_val": torch.stack([item["c_val"] for item in batch], dim=0),
        "rxn_emb": torch.stack([item["rxn_emb"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "target_mask": torch.stack([item["target_mask"] for item in batch], dim=0),
        "cliff_mol": torch.stack([item["cliff_mol"] for item in batch], dim=0),
    }


class ReactionFusionModel(nn.Module):
    def __init__(
        self,
        reaction_model: ReactionEffectNetGINMultiTask,
        num_tasks: int,
        hidden_dim: int = 256,
        reactant_slots: int = 3,
        dropout: float = 0.0,
        freeze_reaction_model: bool = True,
        use_absolute_branch: bool = True,
    ) -> None:
        super().__init__()
        self.reaction_model = reaction_model
        self.num_tasks = num_tasks
        self.freeze_reaction_model = freeze_reaction_model
        self.use_absolute_branch = use_absolute_branch
        self.product_encoder = GINGraphEncoder(hidden_dim=hidden_dim, dropout=dropout)
        hidden_half_dim = hidden_dim // 2
        if use_absolute_branch:
            self.task_gate = nn.Sequential(
                nn.Linear(hidden_dim + hidden_half_dim * 2 + 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.task_gate = None
        fusion_dim = hidden_dim + reactant_slots * hidden_dim + num_tasks * reactant_slots + num_tasks * hidden_half_dim
        if use_absolute_branch:
            fusion_dim += num_tasks
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks),
        )
        if freeze_reaction_model:
            for parameter in self.reaction_model.parameters():
                parameter.requires_grad = False

    def forward(self, batch):
        product_emb = self.product_encoder(batch["product_graph_batch"])
        context = torch.no_grad() if self.freeze_reaction_model else torch.enable_grad()
        with context:
            if self.use_absolute_branch:
                reactant_feature, product_feature = self.reaction_model.encode(
                    batch["reactant_graph_batch"],
                    batch["product_graph_batch"],
                    batch["fp"],
                    batch["val"],
                    batch["c_fp"],
                    batch["rxn_emb"],
                    batch["mask"],
                )
                delta, absolute_prediction, delta_hidden, absolute_hidden = self.reaction_model.predict_all(
                    batch["reactant_graph_batch"],
                    batch["product_graph_batch"],
                    batch["fp"],
                    batch["val"],
                    batch["c_fp"],
                    batch["rxn_emb"],
                    batch["mask"],
                )
            else:
                reactant_feature, delta, delta_hidden = self.reaction_model.predict_delta_only(
                    batch["reactant_graph_batch"],
                    batch["product_graph_batch"],
                    batch["fp"],
                    batch["val"],
                    batch["c_fp"],
                    batch["rxn_emb"],
                    batch["mask"],
                )
            slot_mask = batch["mask"].unsqueeze(-1).to(dtype=reactant_feature.dtype)
            task_slot_mask = slot_mask.unsqueeze(1)
            reactant_feature = reactant_feature * slot_mask
            delta = delta * task_slot_mask
            delta_hidden = delta_hidden * task_slot_mask
            delta_value = delta.squeeze(-1)
            reactant_mask = task_slot_mask
            delta_hidden_task = (delta_hidden * reactant_mask).sum(dim=2) / reactant_mask.sum(dim=2).clamp_min(1.0)
            delta_value_mask = batch["mask"].unsqueeze(1).to(dtype=delta_value.dtype)
            delta_value_task = (delta_value * delta_value_mask).sum(dim=2) / delta_value_mask.sum(dim=2).clamp_min(1.0)
        if self.use_absolute_branch:
            absolute_hidden_task = absolute_hidden.unsqueeze(1).expand(-1, self.num_tasks, -1)
            absolute_value_task = absolute_prediction.view(absolute_prediction.size(0), self.num_tasks, 1)
            product_task = product_emb.unsqueeze(1).expand(-1, self.num_tasks, -1)
            gate_input = torch.cat(
                [
                    product_task,
                    delta_hidden_task,
                    absolute_hidden_task,
                    delta_value_task.unsqueeze(-1),
                    absolute_value_task,
                ],
                dim=-1,
            )
            task_gate = self.task_gate(gate_input)
            fused_task_hidden = task_gate * delta_hidden_task + (1.0 - task_gate) * absolute_hidden_task
        else:
            fused_task_hidden = delta_hidden_task
        feature_parts = [
            product_emb,
            reactant_feature.reshape(reactant_feature.size(0), -1),
            delta_value.reshape(delta_value.size(0), -1),
            fused_task_hidden.reshape(fused_task_hidden.size(0), -1),
        ]
        if self.use_absolute_branch:
            feature_parts.insert(3, absolute_prediction)
        features = torch.cat(feature_parts, dim=1)
        return self.head(features)


def load_reaction_model(spec: DatasetSpec, dataset: ReactionEffectDataset, checkpoint_path: str, config: FusionConfig, device):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"]
    rxn_emb_dim = dataset.rxn_embeddings.shape[1]
    model = ReactionEffectNetGINMultiTask(
        metric_names=spec.metric_names,
        fp_dim=config.fp_dim,
        hidden_dim=config.hidden_dim,
        rxn_emb_dim=rxn_emb_dim,
        num_gin_layers=config.num_gin_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        reactant_slots=config.reactant_slots,
    ).to(device)
    model.load_state_dict(state, strict=False)
    if "metric_mean" in checkpoint and "metric_std" in checkpoint:
        model.set_metric_stats(checkpoint["metric_mean"].to(device), checkpoint["metric_std"].to(device))
    model.eval()
    return model


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def regression_stats(dataset, train_idx):
    values = np.stack([dataset.records[idx]["target"] for idx in train_idx]).astype(np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return torch.from_numpy(mean), torch.from_numpy(std)


def roc_auc_score(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score as sk_roc_auc
        return float(sk_roc_auc(labels, scores))
    except Exception:
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate_predictions(logits, target, mask, spec, target_mean=None, target_std=None, cliff_mol=None):
    if spec.task_type == "regression":
        pred = logits.cpu() * target_std + target_mean
        true = target.cpu()
        valid = mask.cpu().bool()
        scores = {
            "rmse": float(torch.sqrt(((pred - true) ** 2).masked_select(valid).mean()).item()),
            "num_test": int(valid.sum().item()),
        }
        if cliff_mol is not None and pred.size(1) == 1:
            cliff_valid = valid[:, 0] & cliff_mol.cpu().bool()
            scores["num_test_cliff"] = int(cliff_valid.sum().item())
            if bool(cliff_valid.any()):
                scores["cliff_rmse"] = float(
                    torch.sqrt(((pred[:, 0] - true[:, 0]) ** 2)[cliff_valid].mean()).item()
                )
        return scores
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = target.cpu().numpy()
    masks = mask.cpu().numpy().astype(bool)
    scores = {}
    task_scores = []
    for task_idx, task_name in enumerate(spec.target_columns):
        valid = masks[:, task_idx]
        if valid.sum() == 0 or len(np.unique(labels[valid, task_idx])) < 2:
            score = float("nan")
        else:
            score = roc_auc_score(labels[valid, task_idx], probs[valid, task_idx])
            task_scores.append(score)
        scores[f"roc_auc_{task_name}"] = score
    scores["roc_auc"] = float(np.mean(task_scores)) if task_scores else float("nan")
    return scores


def run_epoch(model, loader, optimizer, device, spec, target_mean=None, target_std=None):
    is_train = optimizer is not None
    model.train(is_train)
    all_logits, all_targets, all_masks, all_cliff_labels = [], [], [], []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            logits = model(batch)
            target = batch["target"].float()
            mask = batch["target_mask"].bool()
            if spec.task_type == "regression":
                scaled_target = (target - target_mean.to(device)) / target_std.to(device)
                loss_all = (logits - scaled_target) ** 2
            else:
                loss_all = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = loss_all.masked_select(mask).mean() if bool(mask.any()) else loss_all.sum() * 0.0
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(mask.sum().item())
        total_count += int(mask.sum().item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())
        all_masks.append(mask.detach().cpu())
        all_cliff_labels.append(batch["cliff_mol"].detach().cpu())
    metrics = evaluate_predictions(
        torch.cat(all_logits),
        torch.cat(all_targets),
        torch.cat(all_masks),
        spec,
        target_mean=target_mean,
        target_std=target_std,
        cliff_mol=torch.cat(all_cliff_labels),
    )
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def save_checkpoint(path, epoch, model, optimizer, best_val_score, config, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_score": best_val_score,
            "dataset": spec.name,
            "config": asdict(config),
        },
        path,
    )


def train_fusion(
    spec: DatasetSpec,
    config: FusionConfig,
    split_indices,
    dataset: ReactionEffectDataset,
):
    train_set, val_set, test_set = make_subsets(dataset, split_indices)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_fusion, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fusion, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fusion, pin_memory=torch.cuda.is_available())
    device = torch.device(config.device)
    reaction_model = load_reaction_model(spec, dataset, config.reaction_checkpoint_path, config, device)
    model = build_model_with_seed(
        lambda: ReactionFusionModel(
            reaction_model,
            num_tasks=len(spec.target_columns),
            hidden_dim=config.hidden_dim,
            reactant_slots=config.reactant_slots,
            dropout=config.dropout,
            freeze_reaction_model=config.freeze_reaction_model,
            use_absolute_branch=config.use_absolute_branch,
        ),
        config.model_init_seed,
    ).to(device)
    target_mean = torch.zeros(len(spec.target_columns))
    target_std = torch.ones(len(spec.target_columns))
    if spec.task_type == "regression":
        target_mean, target_std = regression_stats(dataset, split_indices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if spec.task_type == "regression" else "max",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_learning_rate,
    )
    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    best_val = math.inf if spec.task_type == "regression" else -math.inf
    metric_name = "rmse" if spec.task_type == "regression" else "roc_auc"
    print(f"[fusion] {spec.name}: samples={len(dataset)} split={len(train_set)}/{len(val_set)}/{len(test_set)}")
    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, spec, target_mean, target_std)
        val_metrics = run_epoch(model, val_loader, None, device, spec, target_mean, target_std)
        val_score = val_metrics[metric_name]
        if math.isfinite(val_score):
            scheduler.step(val_score)
        print(
            f"[fusion:{spec.name}] epoch {epoch + 1:03d}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_{metric_name}={val_score:.6f}"
        )
        save_checkpoint(latest_path, epoch, model, optimizer, best_val, config, spec)
        improved = val_score < best_val if spec.task_type == "regression" else math.isfinite(val_score) and val_score > best_val
        if improved:
            best_val = val_score
            save_checkpoint(best_path, epoch, model, optimizer, best_val, config, spec)
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    test_metrics = run_epoch(model, test_loader, None, device, spec, target_mean, target_std)
    print(f"Final {spec.name} test_{metric_name}={test_metrics[metric_name]:.6f}")
    if "cliff_rmse" in test_metrics:
        print(
            f"Final {spec.name} test_cliff_rmse={test_metrics['cliff_rmse']:.6f} "
            f"({test_metrics['num_test_cliff']}/{test_metrics['num_test']} cliff molecules)"
        )
    return {
        "dataset": spec.name,
        "metric": metric_name,
        "best_val_score": best_val,
        "test_score": test_metrics[metric_name],
        **(
            {"test_cliff_rmse": test_metrics["cliff_rmse"]}
            if "cliff_rmse" in test_metrics
            else {}
        ),
        **{
            key: value
            for key, value in test_metrics.items()
            if key in {"num_test", "num_test_cliff"}
        },
        **{key: value for key, value in test_metrics.items() if key.startswith("roc_auc_")},
    }
