from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset


def random_scaffold_split(
    smiles_list: Sequence[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.RandomState(seed)
    scaffolds = defaultdict(list)
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
        if mol is None:
            scaffold = f"__invalid_smiles_{index}"
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
            except ValueError:
                scaffold = f"__invalid_scaffold_{index}"
        scaffolds[scaffold].append(index)

    scaffold_values = list(scaffolds.values())
    scaffold_sets = [scaffold_values[index] for index in rng.permutation(len(scaffold_values))]
    val_target = int(np.floor(val_ratio * len(smiles_list)))
    test_target = int(np.floor(test_ratio * len(smiles_list)))
    train_idx, val_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(val_idx) + len(scaffold_set) <= val_target:
            val_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= test_target:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    return train_idx, val_idx, test_idx


def random_split_indices(n_items: int, seed: int, val_ratio: float, test_ratio: float):
    test_size = max(1, int(n_items * test_ratio))
    val_size = max(1, int(n_items * val_ratio))
    train_size = n_items - val_size - test_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for configured split ratios.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_items, generator=generator).tolist()
    return indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]


def official_cliff_split(dataset, seed: int, n_folds: int = 5):
    train_pool = []
    test_idx = []
    for index, record in enumerate(dataset.records):
        split = str(record.get("split", "")).strip().lower()
        if split == "train":
            train_pool.append(index)
        elif split == "test":
            test_idx.append(index)
        else:
            raise ValueError(
                f"{dataset.spec.name} record {record.get('id', index)} is missing "
                "an official split='train' or split='test' label."
            )

    if len(train_pool) < n_folds or not test_idx:
        raise ValueError(
            f"{dataset.spec.name} needs at least {n_folds} official training records "
            "and a non-empty official test split."
        )

    targets = np.asarray(
        [float(dataset.records[index]["target"][0]) for index in train_pool],
        dtype=np.float32,
    )
    median = float(np.median(targets))
    strata = (targets >= median).astype(np.int64)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds = list(splitter.split(strata, strata))
    fold_index = int(seed) % n_folds
    train_pos, val_pos = folds[fold_index]
    train_idx = [train_pool[position] for position in train_pos]
    val_idx = [train_pool[position] for position in val_pos]
    return train_idx, val_idx, test_idx


def make_split_indices(dataset, split_method: str, seed: int, val_ratio: float, test_ratio: float):
    method = split_method.lower()
    if method == "auto":
        method = getattr(dataset.spec, "split_method", "scaffold").lower()
    if method == "random":
        return random_split_indices(len(dataset), seed, val_ratio, test_ratio)
    if method in {"scaffold", "scaffolds", "random_scaffold"}:
        return random_scaffold_split(dataset.product_smiles_list(), seed, val_ratio, test_ratio)
    if method in {"official_cliff", "cliff", "clif"}:
        return official_cliff_split(dataset, seed)
    raise ValueError(f"Unknown split_method: {split_method}")


def make_subsets(dataset, split_indices):
    train_idx, val_idx, test_idx = split_indices
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
