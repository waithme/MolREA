import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch

if __package__ in {None, ""}:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    __package__ = package_dir.name

from .chemistry import mol_from_smiles, morgan_count_fp
from .models.reaction_effect_net_gin_multitask import (
    ReactionEffectNetGINMultiTask,
    dummy_graph,
    mol_to_graph,
    target_delta_normalized,
)


METRIC_NAMES = (
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "MW",
    "RotatableBonds",
    "FractionCSP3",
    "AromaticProportion",
)


@dataclass
class PretrainConfig:
    csv_path: str = "../data/uspto.csv"
    rxnfp_cache_path: str = "../data/processed/uspto_rxnfp_reaction_effect_multitask.npy"
    rxnfp_meta_path: str = "../data/processed/uspto_rxnfp_reaction_effect_multitask_meta.json"
    output_dir: str = "results/pretrain"
    backbone_output_path: str = "pretrain_backbone/best_backbone.pt"
    metric_names: Tuple[str, ...] = METRIC_NAMES
    sample_size: Optional[int] = None
    force_rebuild_rxnfp: bool = False
    seed: int = 0
    batch_size: int = 1024
    num_workers: int = 10
    epochs: int = 100
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 5
    val_ratio: float = 0.1
    reactant_slots: int = 3
    fp_dim: int = 2048
    hidden_dim: int = 256
    num_gin_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    rxnfp_batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(__file__).resolve().parent / candidate).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_product_smiles(product_side: str) -> str:
    products = [part for part in product_side.split(".") if part]
    if not products:
        return ""
    if len(products) == 1:
        return products[0]

    def heavy_atom_count(smiles: str) -> int:
        mol = mol_from_smiles(smiles)
        return mol.GetNumHeavyAtoms() if mol is not None else -1

    return max(products, key=heavy_atom_count)


def parse_reaction_smiles(reaction: str) -> Tuple[list[str], str]:
    if ">>" not in reaction:
        return [], ""
    reactant_side, product_side = reaction.split(">>", 1)
    reactants = [part for part in reactant_side.split(".") if part]
    return reactants, choose_product_smiles(product_side)


def read_reactions(path: Path, sample_size: Optional[int]) -> list[str]:
    reactions = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "reactions" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a 'reactions' column.")
        for row in reader:
            reaction = str(row.get("reactions", "")).strip()
            if reaction and ">>" in reaction:
                reactions.append(reaction)
            if sample_size is not None and len(reactions) >= sample_size:
                break
    if not reactions:
        raise ValueError(f"No valid reaction SMILES found in {path}.")
    return reactions


def reaction_digest(reactions: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for reaction in reactions:
        digest.update(reaction.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def generate_rxnfp_embeddings(reactions: Sequence[str], batch_size: int) -> np.ndarray:
    try:
        from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator
        from rxnfp.models import get_default_model_and_tokenizer
    except ImportError as exc:
        raise ImportError(
            "RXNFP cache is missing and the 'rxnfp' package is not installed. "
            "Install rxnfp or provide the precomputed cache configured by "
            "PretrainConfig.rxnfp_cache_path."
        ) from exc

    model, tokenizer = get_default_model_and_tokenizer()
    generator = RXNBERTFingerprintGenerator(model, tokenizer)
    embeddings = []
    for start in range(0, len(reactions), batch_size):
        embeddings.extend(generator.convert_batch(list(reactions[start:start + batch_size])))
    return np.asarray(embeddings, dtype=np.float32)


def load_or_build_rxnfp(config: PretrainConfig, reactions: Sequence[str]) -> np.ndarray:
    cache_path = resolve_path(config.rxnfp_cache_path)
    meta_path = resolve_path(config.rxnfp_meta_path)
    expected_meta = {
        "num_reactions": len(reactions),
        "sha256": reaction_digest(reactions),
    }
    if cache_path.exists() and meta_path.exists() and not config.force_rebuild_rxnfp:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        embeddings = np.load(cache_path)
        if metadata == expected_meta and embeddings.shape[0] == len(reactions):
            return embeddings.astype(np.float32, copy=False)

    embeddings = generate_rxnfp_embeddings(reactions, config.rxnfp_batch_size)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(reactions):
        raise ValueError(
            f"Invalid RXNFP shape {embeddings.shape}; expected ({len(reactions)}, embedding_dim)."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return embeddings


def descriptor_values(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros(len(METRIC_NAMES), dtype=np.float32)
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    values = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        aromatic_atoms / heavy_atoms if heavy_atoms else 0.0,
    ]
    return np.asarray(values, dtype=np.float32)


class USPTOReactionDataset(Dataset):
    def __init__(
        self,
        reactions: Sequence[str],
        rxn_embeddings: np.ndarray,
        reactant_slots: int,
        fp_dim: int,
    ) -> None:
        if len(reactions) != rxn_embeddings.shape[0]:
            raise ValueError("Reactions and RXNFP embeddings must have the same length.")
        self.reactions = list(reactions)
        self.rxn_embeddings = rxn_embeddings.astype(np.float32, copy=False)
        self.reactant_slots = reactant_slots
        self.fp_dim = fp_dim

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, index: int) -> Dict:
        reactants, product = parse_reaction_smiles(self.reactions[index])
        fp = np.zeros((self.reactant_slots, self.fp_dim), dtype=np.float32)
        values = np.zeros((self.reactant_slots, len(METRIC_NAMES)), dtype=np.float32)
        mask = np.zeros(self.reactant_slots, dtype=bool)
        reactant_graphs = []

        for slot in range(self.reactant_slots):
            if slot < len(reactants):
                smiles = reactants[slot]
                reactant_graphs.append(mol_to_graph(smiles))
                fp[slot] = morgan_count_fp(mol_from_smiles(smiles), self.fp_dim)
                values[slot] = descriptor_values(smiles)
                mask[slot] = True
            else:
                reactant_graphs.append(dummy_graph())

        return {
            "reactant_graphs": reactant_graphs,
            "product_graph": mol_to_graph(product),
            "fp": torch.from_numpy(fp),
            "val": torch.from_numpy(values),
            "c_fp": torch.from_numpy(
                morgan_count_fp(mol_from_smiles(product), self.fp_dim)
            ),
            "c_val": torch.from_numpy(descriptor_values(product)),
            "rxn_emb": torch.from_numpy(self.rxn_embeddings[index]),
            "mask": torch.from_numpy(mask),
        }


def collate_pretrain(batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    reactant_graphs = []
    for item in batch:
        reactant_graphs.extend(item["reactant_graphs"])
    return {
        "reactant_graph_batch": Batch.from_data_list(reactant_graphs),
        "product_graph_batch": Batch.from_data_list(
            [item["product_graph"] for item in batch]
        ),
        "fp": torch.stack([item["fp"] for item in batch]),
        "val": torch.stack([item["val"] for item in batch]),
        "c_fp": torch.stack([item["c_fp"] for item in batch]),
        "c_val": torch.stack([item["c_val"] for item in batch]),
        "rxn_emb": torch.stack([item["rxn_emb"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }


def fixed_train_val_indices(
    num_samples: int, seed: int, val_ratio: float
) -> Tuple[list[int], list[int]]:
    if num_samples < 2:
        raise ValueError("Pretraining requires at least two reactions.")
    val_size = max(1, int(num_samples * val_ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    return indices[val_size:], indices[:val_size]


def compute_metric_stats(
    reactions: Sequence[str], train_indices: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = []
    for index in train_indices:
        reactants, product = parse_reaction_smiles(reactions[index])
        values.extend(descriptor_values(smiles) for smiles in reactants)
        values.append(descriptor_values(product))
    stacked = np.stack(values).astype(np.float32)
    mean = torch.from_numpy(stacked.mean(axis=0))
    std = torch.from_numpy(stacked.std(axis=0))
    return mean, std.clamp_min(1e-8)


def uncertainty_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    log_vars: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slot_mask = mask.bool().unsqueeze(1).unsqueeze(-1)
    task_losses = []
    task_squared_errors = []
    task_counts = []
    for task_idx in range(prediction.size(1)):
        squared_error = (
            prediction[:, task_idx:task_idx + 1]
            - target[:, task_idx:task_idx + 1]
        ).pow(2)
        valid_error = squared_error.masked_select(slot_mask)
        task_losses.append(
            valid_error.mean() if valid_error.numel() else squared_error.sum() * 0.0
        )
        task_squared_errors.append(valid_error.sum().detach())
        task_counts.append(valid_error.numel())
    losses = torch.stack(task_losses)
    weighted = torch.sum(torch.exp(-log_vars) * losses + log_vars)
    return weighted, torch.stack(task_squared_errors), torch.tensor(task_counts)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    squared_error = torch.zeros(model.num_tasks, dtype=torch.float64)
    counts = torch.zeros(model.num_tasks, dtype=torch.long)

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            prediction, _ = model.predict_delta_and_absolute(
                batch["reactant_graph_batch"],
                batch["product_graph_batch"],
                batch["fp"],
                batch["val"],
                batch["c_fp"],
                batch["rxn_emb"],
                batch["mask"],
            )
            target = target_delta_normalized(
                batch["c_val"], batch["val"], model.metric_std
            )
            loss, batch_squared_error, batch_counts = uncertainty_loss(
                prediction, target, batch["mask"], model.log_vars
            )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        squared_error += batch_squared_error.double().cpu()
        counts += batch_counts.long().cpu()

    rmse = torch.sqrt(squared_error / counts.clamp_min(1)).tolist()
    return total_loss / max(total_batches, 1), rmse


def backbone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    excluded = ("prediction_heads.", "absolute_prediction_head.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(excluded) and key != "log_vars"
    }


def save_checkpoint(path, epoch, model, optimizer, best_val_loss, config, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "metric_mean": model.metric_mean.detach().cpu(),
            "metric_std": model.metric_std.detach().cpu(),
            "metric_names": list(model.metric_names),
            "config": asdict(config),
            "split": split,
        },
        path,
    )


def save_backbone(path, epoch, model, best_val_loss, config, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "backbone_state_dict": backbone_state_dict(model),
            "best_val_loss": best_val_loss,
            "metric_mean": model.metric_mean.detach().cpu(),
            "metric_std": model.metric_std.detach().cpu(),
            "metric_names": list(model.metric_names),
            "config": asdict(config),
            "split": split,
        },
        path,
    )


def build_dataloaders(config: PretrainConfig):
    reactions = read_reactions(resolve_path(config.csv_path), config.sample_size)
    embeddings = load_or_build_rxnfp(config, reactions)
    dataset = USPTOReactionDataset(
        reactions, embeddings, config.reactant_slots, config.fp_dim
    )
    train_idx, val_idx = fixed_train_val_indices(
        len(dataset), config.seed, config.val_ratio
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_pretrain,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_pretrain,
        pin_memory=torch.cuda.is_available(),
    )
    return reactions, embeddings.shape[1], train_idx, val_idx, train_loader, val_loader


def train(config: PretrainConfig) -> dict:
    if tuple(config.metric_names) != METRIC_NAMES:
        raise ValueError(
            "Paper-aligned pretraining requires exactly these eight descriptors: "
            f"{METRIC_NAMES}"
        )
    if config.val_ratio != 0.1:
        raise ValueError("Paper-aligned pretraining requires a fixed 9:1 train/validation split.")
    set_seed(config.seed)
    (
        reactions,
        rxn_emb_dim,
        train_idx,
        val_idx,
        train_loader,
        val_loader,
    ) = build_dataloaders(config)
    metric_mean, metric_std = compute_metric_stats(reactions, train_idx)
    device = torch.device(config.device)
    model = ReactionEffectNetGINMultiTask(
        metric_names=config.metric_names,
        fp_dim=config.fp_dim,
        hidden_dim=config.hidden_dim,
        rxn_emb_dim=rxn_emb_dim,
        num_gin_layers=config.num_gin_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        reactant_slots=config.reactant_slots,
    ).to(device)
    model.set_metric_stats(metric_mean.to(device), metric_std.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_learning_rate,
    )

    output_dir = resolve_path(config.output_dir)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    backbone_path = resolve_path(config.backbone_output_path)
    split_metadata = {
        "method": "fixed_random_9_to_1",
        "seed": config.seed,
        "num_train": len(train_idx),
        "num_validation": len(val_idx),
        "num_test": 0,
        "train_indices_sha256": hashlib.sha256(
            np.asarray(train_idx, dtype=np.int64).tobytes()
        ).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(
            np.asarray(val_idx, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(
        json.dumps(split_metadata, indent=2), encoding="utf-8"
    )

    best_val_loss = math.inf
    best_epoch = -1
    print(
        f"[pretrain] samples={len(reactions)} split={len(train_idx)}/{len(val_idx)} "
        f"batch_size={config.batch_size} epochs={config.epochs} dropout={config.dropout}"
    )
    for epoch in range(config.epochs):
        train_loss, train_rmse = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse = run_epoch(model, val_loader, None, device)
        scheduler.step(val_loss)
        print(
            f"[pretrain] epoch {epoch + 1:03d}/{config.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_rmse={train_rmse} val_rmse={val_rmse} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )
        save_checkpoint(
            latest_path, epoch, model, optimizer, best_val_loss, config, split_metadata
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                best_path, epoch, model, optimizer, best_val_loss, config, split_metadata
            )
            save_backbone(
                backbone_path, epoch, model, best_val_loss, config, split_metadata
            )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": str(best_path),
        "backbone": str(backbone_path),
    }


def main() -> None:
    train(PretrainConfig())


if __name__ == "__main__":
    main()
