import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .data.dataset_specs import DatasetSpec, resolve_path
from .data.reaction_dataset import ReactionEffectDataset
from .data.reaction_io import build_reaction_records, load_rows, load_rxn_embeddings
from .data.splits import make_split_indices, make_subsets
from .models.reaction_effect_net_gin_multitask import (
    ReactionEffectNetGINMultiTask,
    load_backbone_weights,
    target_delta_normalized,
)
from .proxy import create_proxy_estimator


@dataclass
class ReactionFinetuneConfig:
    pretrained_backbone_path: str = "pretrain_backbone/best_backbone.pt"
    output_dir: str = "results/reaction_finetune"
    rxnfp_json_path: Optional[str] = None
    sample_size: Optional[int] = None
    model_init_seed: int = 0
    load_pretrained_backbone: bool = True
    seed: int = 0
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 50
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    split_method: str = "scaffold"
    val_ratio: float = 0.09
    test_ratio: float = 0.1
    reactant_slots: int = 3
    fp_dim: int = 2048
    hidden_dim: int = 256
    num_gin_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    proxy_epochs: int = 500
    proxy_batch_size: int = 64
    proxy_learning_rate: float = 1e-3
    proxy_weight_decay: float = 1e-4
    use_absolute_branch: bool = True
    absolute_loss_weight: float = 1.0


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


def build_dataset(spec: DatasetSpec, config: ReactionFinetuneConfig) -> Tuple[ReactionEffectDataset, Tuple[list, list, list]]:
    rows = load_rows(spec, config.sample_size)
    records = build_reaction_records(spec, rows, config.reactant_slots)
    rxnfp_json_path = config.rxnfp_json_path or spec.rxnfp_json_path
    rxn_embeddings = load_rxn_embeddings(
        records,
        rxnfp_json_path=rxnfp_json_path,
    )
    dataset = ReactionEffectDataset(
        spec,
        records,
        rxn_embeddings,
        reactant_slots=config.reactant_slots,
        fp_dim=config.fp_dim,
    )
    split = make_split_indices(dataset, config.split_method, config.seed, config.val_ratio, config.test_ratio)
    train_rows = []
    for idx in split[0]:
        record = records[idx]
        train_rows.append(
            {
                spec.smiles_column: record["product_smiles"],
                **{col: str(record["target"][col_idx]) for col_idx, col in enumerate(spec.target_columns)},
            }
        )
    print(f"[proxy] {spec.name} reaction uses three-layer MLP", flush=True)
    proxy = create_proxy_estimator(
        spec,
        seed=config.seed,
        model_init_seed=config.model_init_seed,
        epochs=config.proxy_epochs,
        batch_size=config.proxy_batch_size,
        learning_rate=config.proxy_learning_rate,
        weight_decay=config.proxy_weight_decay,
    )
    proxy.fit(train_rows)
    dataset.set_value_estimator(proxy)
    return dataset, split


def compute_metric_stats(dataset, train_indices, num_tasks):
    values = []
    for index in train_indices:
        item = dataset[index]
        val = item["val"][item["mask"].bool()]
        if val.numel() > 0:
            values.append(val.float())
        target_mask = item["target_mask"].bool()
        c_val = item["c_val"].view(1, num_tasks).float()
        c_val = torch.where(target_mask.view(1, num_tasks), c_val, torch.full_like(c_val, float("nan")))
        values.append(c_val)
    all_values = torch.cat(values, dim=0)
    valid = torch.isfinite(all_values)
    counts = valid.sum(dim=0).clamp_min(1)
    safe_values = torch.where(valid, all_values, torch.zeros_like(all_values))
    mean = safe_values.sum(dim=0) / counts
    centered = torch.where(valid, all_values - mean.view(1, num_tasks), torch.zeros_like(all_values))
    std = torch.sqrt(centered.pow(2).sum(dim=0) / counts)
    mean = torch.where(valid.any(dim=0), mean, torch.zeros_like(mean))
    std = torch.where(valid.any(dim=0), std, torch.ones_like(std)).clamp_min(1e-8)
    return mean, std


def collate_reaction_effect(batch):
    from torch_geometric.data import Batch

    reactant_graphs = []
    for item in batch:
        reactant_graphs.extend(item["reactant_graphs"])
    product_graphs = [item["product_graph"] for item in batch]
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
    }


def masked_delta_uncertainty_loss(prediction, target, reactant_mask, target_mask, log_vars):
    combined_mask = reactant_mask.bool().unsqueeze(1).unsqueeze(-1)
    combined_mask = combined_mask & target_mask.bool().unsqueeze(-1).unsqueeze(-1)
    task_losses = []
    for task_idx in range(prediction.size(1)):
        squared_error = (prediction[:, task_idx:task_idx + 1] - target[:, task_idx:task_idx + 1]) ** 2
        masked_error = squared_error.masked_select(combined_mask[:, task_idx:task_idx + 1])
        loss = masked_error.mean() if masked_error.numel() else squared_error.sum() * 0.0
        task_losses.append(loss)
    task_losses = torch.stack(task_losses)
    return torch.sum(torch.exp(-log_vars) * task_losses + log_vars), combined_mask


def masked_absolute_mse_loss(prediction, target, target_mask):
    mask = target_mask.bool()
    squared_error = (prediction - target) ** 2
    masked_error = squared_error.masked_select(mask)
    if masked_error.numel() == 0:
        return squared_error.sum() * 0.0
    return masked_error.mean()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(model, loader, optimizer, device, spec, use_absolute_branch, absolute_loss_weight):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    total_sq = 0.0
    total_slots = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            if use_absolute_branch:
                prediction, absolute_prediction = model.predict_delta_and_absolute(
                    batch["reactant_graph_batch"],
                    batch["product_graph_batch"],
                    batch["fp"],
                    batch["val"],
                    batch["c_fp"],
                    batch["rxn_emb"],
                    batch["mask"],
                )
            else:
                _, prediction, _ = model.predict_delta_only(
                    batch["reactant_graph_batch"],
                    batch["product_graph_batch"],
                    batch["fp"],
                    batch["val"],
                    batch["c_fp"],
                    batch["rxn_emb"],
                    batch["mask"],
                )
            target = target_delta_normalized(batch["c_val"], batch["val"], model.metric_std)
            delta_loss, combined_mask = masked_delta_uncertainty_loss(
                prediction,
                target,
                batch["mask"],
                batch["target_mask"],
                model.log_vars,
            )
            loss = delta_loss
            if use_absolute_branch:
                if spec.task_type == "regression":
                    absolute_target = (
                        batch["target"].float() - model.metric_mean.view(1, -1)
                    ) / model.metric_std.view(1, -1)
                else:
                    absolute_target = batch["target"].float()
                absolute_loss = masked_absolute_mse_loss(
                    absolute_prediction,
                    absolute_target,
                    batch["target_mask"],
                )
                loss = loss + absolute_loss_weight * absolute_loss
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        raw_pred = prediction * model.metric_std.view(1, -1, 1, 1)
        raw_target = (batch["c_val"].unsqueeze(1) - batch["val"]).transpose(1, 2).unsqueeze(-1)
        total_sq += float(((raw_pred - raw_target) ** 2).masked_select(combined_mask).sum().detach().cpu())
        total_slots += int(combined_mask.sum().item())
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1), math.sqrt(total_sq / max(total_slots, 1))


def save_checkpoint(path, epoch, model, optimizer, best_val_rmse, config, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_val_rmse,
            "metric_mean": model.metric_mean.detach().cpu(),
            "metric_std": model.metric_std.detach().cpu(),
            "metric_names": list(model.metric_names),
            "dataset": spec.name,
            "config": asdict(config),
        },
        path,
    )


def train_reaction_effect(
    spec: DatasetSpec,
    config: ReactionFinetuneConfig,
    dataset: Optional[ReactionEffectDataset] = None,
    split_indices=None,
):
    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset is None:
        dataset, built_split = build_dataset(spec, config)
        split_indices = split_indices or built_split
    elif split_indices is None:
        split_indices = make_split_indices(
            dataset,
            config.split_method,
            config.seed,
            config.val_ratio,
            config.test_ratio,
        )
    train_set, val_set, test_set = make_subsets(dataset, split_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_reaction_effect,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_reaction_effect,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_reaction_effect,
    )
    device = torch.device(config.device)
    rxn_emb_dim = dataset.rxn_embeddings.shape[1]
    model = build_model_with_seed(
        lambda: ReactionEffectNetGINMultiTask(
            metric_names=spec.metric_names,
            fp_dim=config.fp_dim,
            hidden_dim=config.hidden_dim,
            rxn_emb_dim=rxn_emb_dim,
            num_gin_layers=config.num_gin_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            reactant_slots=config.reactant_slots,
        ),
        config.model_init_seed,
    ).to(device)
    metric_mean, metric_std = compute_metric_stats(dataset, split_indices[0], len(spec.metric_names))
    model.set_metric_stats(metric_mean.to(device), metric_std.to(device))
    backbone_path = resolve_path(config.pretrained_backbone_path)
    if config.load_pretrained_backbone and backbone_path.exists():
        load_backbone_weights(model, backbone_path, device)
    optimizer_groups = [
        {"params": model.prediction_heads.parameters(), "lr": config.learning_rate},
        {"params": model.log_vars, "lr": config.learning_rate},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("prediction_heads") and not n.startswith("absolute_prediction_head") and n != "log_vars"], "lr": config.backbone_learning_rate},
    ]
    if config.use_absolute_branch:
        optimizer_groups.insert(
            1,
            {"params": model.absolute_prediction_head.parameters(), "lr": config.learning_rate},
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config.weight_decay)
    best_val_rmse = math.inf
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    print(f"[reaction] {spec.name}: samples={len(dataset)} split={len(train_set)}/{len(val_set)}/{len(test_set)}")
    for epoch in range(config.epochs):
        train_loss, train_rmse = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            spec,
            config.use_absolute_branch,
            config.absolute_loss_weight,
        )
        val_loss, val_rmse = run_epoch(
            model,
            val_loader,
            None,
            device,
            spec,
            config.use_absolute_branch,
            config.absolute_loss_weight,
        )
        print(
            f"[reaction:{spec.name}] epoch {epoch + 1:03d}/{config.epochs} "
            f"train_loss={train_loss:.6f} train_delta_rmse={train_rmse:.6f} "
            f"val_loss={val_loss:.6f} val_delta_rmse={val_rmse:.6f}"
        )
        save_checkpoint(latest_path, epoch, model, optimizer, best_val_rmse, config, spec)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(best_path, epoch, model, optimizer, best_val_rmse, config, spec)
            print(f"  saved reaction best: val_delta_rmse={best_val_rmse:.6f}")

    selected_path = best_path if best_path.exists() else latest_path
    checkpoint = torch.load(selected_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, best_test_rmse = run_epoch(
        model,
        test_loader,
        None,
        device,
        spec,
        config.use_absolute_branch,
        config.absolute_loss_weight,
    )
    print(f"Final {spec.name} test_delta_rmse={best_test_rmse:.6f}")
    return {
        "dataset": spec.name,
        "best_val_rmse": best_val_rmse,
        "best_test_rmse": best_test_rmse,
        "checkpoint": str(best_path),
        "split_indices": split_indices,
    }
