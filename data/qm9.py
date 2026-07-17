import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch import nn
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch, Data

from ..chemistry import mol_from_smiles, morgan_count_fp
from ..models.reaction_effect_net_gin_multitask import dummy_graph, mol_to_graph
from .dataset_specs import PROJECT_ROOT, resolve_path
from .splits import random_scaffold_split, random_split_indices


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

TARGET_COLUMNS = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
)
QM9_EXPERIMENT_TARGETS = (
    "mu",
    "alpha",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
)
DEFAULT_MODEL_NAMES = ("gcn", "gin", "graphsage", "gat", "dmpnn", "morgan_mlp")
DEFAULT_REACTANT_SLOTS = 3

PathLike = Union[str, Path]
_GRAPH_CACHE: Dict[Tuple[str, str, Optional[int]], Tuple[list, list]] = {}
_REACTION_JSON_CACHE: Dict[str, Dict] = {}
_RXNFP_JSON_CACHE: Dict[str, object] = {}
_SMILES_GRAPH_CACHE: Dict[str, Data] = {}
_SMILES_FP_CACHE: Dict[Tuple[str, int], np.ndarray] = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_existing(path: PathLike) -> Path:
    path = Path(path)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, resolve_path(path)]
    resolved = candidates[0]
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break
    if not resolved.exists():
        raise FileNotFoundError(f"Missing QM9 file: {resolved}")
    return resolved


def load_descriptor_map(path: PathLike) -> Dict[str, List[str]]:
    resolved = _resolve_existing(path)
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{resolved} must be a JSON object.")
        return {
            str(target): [str(descriptor) for descriptor in descriptors]
            for target, descriptors in payload.items()
            if isinstance(descriptors, list)
        }

    descriptor_map: Dict[str, List[str]] = {}
    with resolved.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row["Target_Metric"].strip()
            descriptor = row["Descriptor"].strip()
            descriptor_map.setdefault(target, [])
            if descriptor not in descriptor_map[target]:
                descriptor_map[target].append(descriptor)
    return descriptor_map


_DESCRIPTOR_FUNCS = dict(Descriptors.descList)
_DESCRIPTOR_FUNCS_CASEFOLD = {name.casefold(): func for name, func in _DESCRIPTOR_FUNCS.items()}
_DESCRIPTOR_ALIASES = {
    "rotatable_bond_count": lambda mol: float(Descriptors.NumRotatableBonds(mol)),
    "phi": lambda mol: float(rdMolDescriptors.CalcPhi(mol)),
}


def descriptor_value(mol: Chem.Mol, descriptor_name: str) -> float:
    name = descriptor_name.strip()
    func = _DESCRIPTOR_ALIASES.get(name.casefold()) or _DESCRIPTOR_FUNCS.get(name)
    if func is None:
        func = _DESCRIPTOR_FUNCS_CASEFOLD.get(name.casefold())
    if func is None:
        raise KeyError(f"Unsupported RDKit descriptor: {descriptor_name}")
    try:
        value = float(func(mol))
    except Exception:
        value = 0.0
    return value if math.isfinite(value) else 0.0


def compute_descriptor_vector(smiles: str, descriptor_names: Sequence[str]) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(len(descriptor_names), dtype=np.float32)
    values = [descriptor_value(mol, name) for name in descriptor_names]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class QM9Record:
    cid: str
    smiles: str
    targets: Dict[str, float]


def load_qm9_records(csv_path: PathLike, sample_size: Optional[int] = None) -> List[QM9Record]:
    records: List[QM9Record] = []
    csv_path = _resolve_existing(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        cid_column = "CID" if "CID" in fieldnames else "mol_id" if "mol_id" in fieldnames else None
        smiles_column = "SMILES" if "SMILES" in fieldnames else "smiles" if "smiles" in fieldnames else None
        if cid_column is None or smiles_column is None:
            raise ValueError(f"{csv_path} must contain CID/mol_id and SMILES/smiles columns.")
        for row_number, row in enumerate(reader, start=2):
            cid = str(row.get(cid_column, "")).strip()
            smiles = str(row.get(smiles_column, "")).strip()
            if cid in {"", "."} or smiles in {"", "."}:
                continue
            targets = {}
            for name in TARGET_COLUMNS:
                raw = row.get(name)
                if raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if math.isfinite(value):
                    targets[name] = value
            if not targets:
                continue
            records.append(
                QM9Record(
                    cid=cid,
                    smiles=smiles,
                    targets=targets,
                )
            )
            if sample_size is not None and len(records) >= sample_size:
                break
    if not records:
        raise ValueError(f"{csv_path} did not contain any usable QM9 records.")
    return records


def load_sdf_molecules_by_cid(sdf_path: PathLike, allowed_cids: Optional[Iterable[str]] = None) -> Dict[str, Chem.Mol]:
    sdf_path = _resolve_existing(sdf_path)
    allowed = set(allowed_cids) if allowed_cids is not None else None
    molecules: Dict[str, Chem.Mol] = {}
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for mol in supplier:
        if mol is None or not mol.HasProp("_Name"):
            continue
        cid = mol.GetProp("_Name")
        if allowed is not None and cid not in allowed:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        molecules[cid] = mol
        if allowed is not None and len(molecules) >= len(allowed):
            break
    return molecules


def mol_from_qm9_smiles(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return None
    return Chem.Mol(mol)


_BOND_TYPE_TO_INT = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


def mol_to_qm9_graph(mol: Chem.Mol, y: Optional[float] = None) -> Data:
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32) if mol.GetNumConformers() else torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float32)
    edge_pairs = []
    edge_attrs = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        attr = [
            _BOND_TYPE_TO_INT.get(bond.GetBondType(), 0),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo()),
        ]
        edge_pairs.extend([(begin, end), (end, begin)])
        edge_attrs.extend([attr, attr])
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long) if edge_attrs else torch.empty((0, 4), dtype=torch.long)
    data = Data(z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    fp = np.zeros(2048, dtype=np.float32)
    sparse_fp = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=2048, useChirality=True)
    for bit_idx, count in sparse_fp.GetNonzeroElements().items():
        fp[bit_idx] = float(count)
    data.fp = torch.from_numpy(fp).view(1, -1)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


class QM9PropertyDataset(Dataset):
    def __init__(
        self,
        csv_path: PathLike,
        sdf_path: Optional[PathLike],
        target_metric: str,
        sample_size: Optional[int] = None,
    ) -> None:
        if target_metric not in TARGET_COLUMNS:
            raise ValueError(f"Unknown QM9 target: {target_metric}. Available: {TARGET_COLUMNS}")
        csv_path = _resolve_existing(csv_path)
        resolved_sdf_path = None
        if sdf_path not in (None, ""):
            candidate = Path(sdf_path)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            if candidate.exists():
                resolved_sdf_path = candidate
            else:
                print(f"[qm9] SDF not found at {candidate}; falling back to SMILES-only 2D graphs.")
        cache_key = (str(csv_path.resolve()), str(resolved_sdf_path.resolve()) if resolved_sdf_path else "smiles_only", sample_size)
        if cache_key not in _GRAPH_CACHE:
            records = load_qm9_records(csv_path, sample_size=sample_size)
            molecules = (
                load_sdf_molecules_by_cid(resolved_sdf_path, allowed_cids=[record.cid for record in records])
                if resolved_sdf_path is not None
                else {}
            )
            items = []
            graphs = []
            for record in records:
                mol = molecules.get(record.cid) or mol_from_qm9_smiles(record.smiles)
                if mol is None:
                    continue
                items.append((record.cid, record.smiles, record.targets))
                graphs.append(mol_to_qm9_graph(mol))
            _GRAPH_CACHE[cache_key] = (items, graphs)
            source = "SDF" if resolved_sdf_path is not None else "SMILES"
            print(f"[qm9] cached product graphs from {source}: {len(graphs)} molecules")
        items, graphs = _GRAPH_CACHE[cache_key]
        self.items: List[Tuple[str, str, float]] = []
        self.graphs: List[Data] = []
        for (cid, smiles, targets), graph in zip(items, graphs):
            if target_metric not in targets:
                available = ", ".join(sorted(targets)) or "none"
                raise KeyError(
                    f"Target {target_metric!r} is missing for CID {cid!r}. "
                    f"Available targets on this row: {available}. "
                    "Check refined_QM9.csv or run with a target column that exists in that file."
                )
            value = float(targets[target_metric])
            target_graph = copy.copy(graph)
            target_graph.y = torch.tensor([value], dtype=torch.float32)
            self.items.append((cid, smiles, value))
            self.graphs.append(target_graph)
        self.target_metric = target_metric

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]


class DescriptorProxyRegressor:
    def __init__(
        self,
        descriptor_names: Sequence[str],
        hidden_dims: Tuple[int, int] = (128, 64),
        epochs: int = 500,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        internal_val_ratio: float = 0.1,
        lr_factor: float = 0.5,
        lr_patience: int = 10,
        min_learning_rate: float = 1e-5,
        early_stop_patience: int = 30,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.descriptor_names = tuple(descriptor_names)
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.internal_val_ratio = internal_val_ratio
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_learning_rate = min_learning_rate
        self.early_stop_patience = early_stop_patience
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.model: Optional[nn.Module] = None
        self._cache: Dict[str, float] = {}

    def _build_model(self) -> nn.Module:
        first_hidden, second_hidden = self.hidden_dims
        return nn.Sequential(
            nn.Linear(len(self.descriptor_names), first_hidden),
            nn.LayerNorm(first_hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(second_hidden, 1),
        ).to(self.device)

    def fit(self, smiles_list: Sequence[str], labels: Sequence[float]) -> "DescriptorProxyRegressor":
        x = np.stack([compute_descriptor_vector(smiles, self.descriptor_names) for smiles in smiles_list])
        y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if float(y.std()) >= 1e-8 else 1.0
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(x), generator=generator)
        val_size = max(1, int(round(len(x) * self.internal_val_ratio))) if len(x) > 1 else 1
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        if train_idx.numel() == 0:
            train_idx = val_idx

        model = self._build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_learning_rate,
        )
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y.astype(np.float32)).to(self.device)
        best_state = copy.deepcopy(model.state_dict())
        best_val = math.inf
        epochs_without_improvement = 0

        for _ in range(self.epochs):
            model.train()
            pred = model(x_tensor[train_idx])
            loss = torch.nn.functional.mse_loss(pred, y_tensor[train_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = torch.nn.functional.mse_loss(model(x_tensor[val_idx]), y_tensor[val_idx])
            val_score = float(val_loss.detach().cpu())
            scheduler.step(val_score)
            if val_score < best_val:
                best_val = val_score
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stop_patience:
                    break

        model.load_state_dict(best_state)
        self.model = model.eval()
        return self

    def predict_one(self, smiles: str) -> float:
        if self.model is None or self.x_mean is None or self.x_std is None:
            return 0.0
        if smiles in self._cache:
            return self._cache[smiles]
        x = compute_descriptor_vector(smiles, self.descriptor_names)
        x = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            pred = self.model(torch.from_numpy(x.astype(np.float32)).view(1, -1).to(self.device))
        value = float(pred.item() * self.y_std + self.y_mean)
        self._cache[smiles] = value
        return value


def fit_proxy_from_subset(
    subset: Subset,
    descriptor_names: Sequence[str],
    device: str,
    seed: int = 0,
) -> DescriptorProxyRegressor:
    smiles = []
    labels = []
    for index in subset.indices:
        _, product_smiles, label = subset.dataset.items[index]
        smiles.append(product_smiles)
        labels.append(float(label))
    return DescriptorProxyRegressor(descriptor_names=descriptor_names, device=device, seed=seed).fit(smiles, labels)


def subset_target_stats(subset: Subset) -> Tuple[float, float]:
    values = [float(subset.dataset[index].y.view(-1)[0].item()) for index in subset.indices]
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, std if std >= 1e-8 else 1.0


def make_qm9_split(dataset: QM9PropertyDataset, method: str, seed: int, val_ratio: float, test_ratio: float):
    method = method.lower()
    if method == "random":
        return random_split_indices(len(dataset), seed, val_ratio, test_ratio)
    if method in {"scaffold", "scaffolds", "random_scaffold"}:
        return random_scaffold_split([smiles for _, smiles, _ in dataset.items], seed, val_ratio, test_ratio)
    raise ValueError(f"Unknown QM9 split method: {method}")


def _normal_key(value) -> str:
    return str(value).strip()


def _load_json_cached(path: PathLike, cache: Dict[str, object]):
    resolved = _resolve_existing(path)
    key = str(resolved.resolve())
    if key not in cache:
        with resolved.open("r", encoding="utf-8") as handle:
            cache[key] = json.load(handle)
    return cache[key]


def _rank_value(meta: Dict) -> int:
    try:
        return int(meta.get("expansion_rank", meta.get("policy_probability_rank", 999999)))
    except Exception:
        return 999999


def select_ranked_reactants(entry) -> List[str]:
    if not isinstance(entry, list):
        return []
    candidates = []
    for item in entry:
        if not isinstance(item, list) or len(item) < 2:
            continue
        reactants, meta = item[0], item[1]
        if isinstance(reactants, list):
            candidates.append((_rank_value(meta if isinstance(meta, dict) else {}), [smiles for smiles in reactants if smiles]))
    candidates = [(rank, smiles) for rank, smiles in candidates if smiles]
    return min(candidates, key=lambda item: item[0])[1] if candidates else []


def _cached_graph(smiles: str) -> Data:
    key = _normal_key(smiles)
    if key not in _SMILES_GRAPH_CACHE:
        _SMILES_GRAPH_CACHE[key] = mol_to_graph(key)
    return copy.copy(_SMILES_GRAPH_CACHE[key])


def _cached_fp(smiles: str, fp_dim: int) -> np.ndarray:
    key = (_normal_key(smiles), int(fp_dim))
    if key not in _SMILES_FP_CACHE:
        _SMILES_FP_CACHE[key] = morgan_count_fp(mol_from_smiles(key[0]), n_bits=fp_dim)
    return _SMILES_FP_CACHE[key]


class QM9ReactionDataset(Dataset):
    def __init__(
        self,
        property_dataset: QM9PropertyDataset,
        reaction_json_path: PathLike,
        rxn_embeddings: np.ndarray,
        reactant_slots: int = DEFAULT_REACTANT_SLOTS,
        fp_dim: int = 2048,
    ) -> None:
        self.property_dataset = property_dataset
        self.reactant_slots = reactant_slots
        self.fp_dim = fp_dim
        self.reaction_data = _load_json_cached(reaction_json_path, _REACTION_JSON_CACHE)
        self.rxn_embeddings = rxn_embeddings.astype(np.float32, copy=False)
        self.items = self._build_items()
        self.value_estimator: Optional[DescriptorProxyRegressor] = None
        if len(self.items) != self.rxn_embeddings.shape[0]:
            raise ValueError("QM9 items and RXNFP embeddings must have the same length.")

    def _build_items(self) -> List[Dict]:
        items = []
        for index, (cid, product_smiles, label) in enumerate(self.property_dataset.items):
            entry = self.reaction_data.get(_normal_key(cid), self.reaction_data.get(_normal_key(product_smiles)))
            reactants = select_ranked_reactants(entry)[: self.reactant_slots]
            items.append(
                {
                    "index": index,
                    "cid": cid,
                    "product_smiles": product_smiles,
                    "label": float(label),
                    "reactants": reactants,
                    "reaction_smiles": ".".join(reactants) + ">>" + product_smiles,
                }
            )
        return items

    def set_value_estimator(self, estimator: DescriptorProxyRegressor) -> None:
        self.value_estimator = estimator

    def __len__(self) -> int:
        return len(self.items)

    def _reactant_value(self, smiles: str) -> float:
        return 0.0 if self.value_estimator is None else self.value_estimator.predict_one(smiles)

    def __getitem__(self, index: int) -> Dict:
        item = self.items[index]
        reactant_graphs = []
        fp = np.zeros((self.reactant_slots, self.fp_dim), dtype=np.float32)
        val = np.zeros((self.reactant_slots, 1), dtype=np.float32)
        mask = np.zeros(self.reactant_slots, dtype=bool)
        for slot in range(self.reactant_slots):
            if slot < len(item["reactants"]):
                smiles = item["reactants"][slot]
                reactant_graphs.append(_cached_graph(smiles))
                fp[slot] = _cached_fp(smiles, self.fp_dim)
                val[slot, 0] = self._reactant_value(smiles)
                mask[slot] = True
            else:
                reactant_graphs.append(dummy_graph())
        product_smiles = item["product_smiles"]
        return {
            "reactant_graphs": reactant_graphs,
            "product_graph": _cached_graph(product_smiles),
            "target_graph": self.property_dataset[index],
            "fp": torch.from_numpy(fp),
            "val": torch.from_numpy(val),
            "c_fp": torch.from_numpy(_cached_fp(product_smiles, self.fp_dim)),
            "c_val": torch.tensor([item["label"]], dtype=torch.float32),
            "rxn_emb": torch.from_numpy(self.rxn_embeddings[index]),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor([item["label"]], dtype=torch.float32),
            "target": torch.tensor([item["label"]], dtype=torch.float32),
            "target_mask": torch.ones(1, dtype=torch.float32),
        }


def _rxnfp_payload(path: PathLike):
    path = Path(path)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.exists() and "rnxfp" in resolved.name:
        alt = resolved.with_name(resolved.name.replace("rnxfp", "rxnfp"))
        if alt.exists():
            resolved = alt
    if not resolved.exists() and "rxnfp" in resolved.name:
        alt = resolved.with_name(resolved.name.replace("rxnfp", "rnxfp"))
        if alt.exists():
            resolved = alt
    if not resolved.exists():
        raise FileNotFoundError(f"Missing QM9 RXNFP file: {resolved}")
    return _load_json_cached(resolved, _RXNFP_JSON_CACHE)


def _rxnfp_vector(entry) -> np.ndarray:
    if isinstance(entry, dict):
        for key in ("rxnfp", "embedding", "vector", "features"):
            if key in entry:
                entry = entry[key]
                break
    vector = np.asarray(entry, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"RXNFP entry must be 1D, got shape {vector.shape}")
    return vector


def _nested_payload(payload):
    if isinstance(payload, dict):
        for key in ("embeddings", "rxnfp", "data", "items", "records"):
            nested = payload.get(key)
            if isinstance(nested, (dict, list)):
                return nested
    return payload


def _numeric_suffix(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _candidate_rxnfp_keys(item: Dict) -> Tuple[str, ...]:
    cid = _normal_key(item.get("cid", ""))
    index = int(item.get("index", 0))
    numeric = _numeric_suffix(cid)
    keys = [
        cid,
        numeric,
        f"gdb_{numeric}" if numeric else "",
        _normal_key(item.get("product_smiles", "")),
        _normal_key(item.get("reaction_smiles", "")),
        str(index),
        str(index + 1),
    ]
    ordered = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


def _lookup_rxnfp_entry(payload, item: Dict):
    payload = _nested_payload(payload)
    if isinstance(payload, dict):
        for key in _candidate_rxnfp_keys(item):
            if key in payload:
                return payload[key]
        return None
    if isinstance(payload, list):
        index = int(item.get("index", -1))
        if 0 <= index < len(payload):
            return payload[index]
    return None


def load_rxnfp_for_qm9_items(items: Sequence[Dict], rxnfp_json_path: PathLike) -> np.ndarray:
    payload = _rxnfp_payload(rxnfp_json_path)
    embeddings: List[Optional[np.ndarray]] = []
    embedding_dim = None
    missing = 0
    for item in items:
        entry = _lookup_rxnfp_entry(payload, item)
        if entry is None:
            embeddings.append(None)
            missing += 1
            continue
        vector = _rxnfp_vector(entry)
        embedding_dim = int(vector.shape[0]) if embedding_dim is None else embedding_dim
        embeddings.append(vector)
    if embedding_dim is None:
        raise ValueError(f"{rxnfp_json_path} does not contain usable RXNFP embeddings.")
    zero = np.zeros(embedding_dim, dtype=np.float32)
    if missing:
        print(f"[qm9] RXNFP missing for {missing} records; using zero embeddings.")
    return np.asarray([zero.copy() if vector is None else vector for vector in embeddings], dtype=np.float32)


def build_qm9_reaction_dataset(
    csv_path: PathLike,
    sdf_path: PathLike,
    reaction_json_path: PathLike,
    rxnfp_json_path: PathLike,
    target_metric: str,
    sample_size: Optional[int],
    reactant_slots: int,
) -> QM9ReactionDataset:
    property_dataset = QM9PropertyDataset(csv_path, sdf_path, target_metric, sample_size=sample_size)
    empty = np.zeros((len(property_dataset), 1), dtype=np.float32)
    probe = QM9ReactionDataset(property_dataset, reaction_json_path, empty, reactant_slots=reactant_slots)
    rxn_embeddings = load_rxnfp_for_qm9_items(probe.items, rxnfp_json_path)
    return QM9ReactionDataset(property_dataset, reaction_json_path, rxn_embeddings, reactant_slots=reactant_slots)


def collate_qm9_fusion(batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    reactant_graphs = []
    for item in batch:
        reactant_graphs.extend(item["reactant_graphs"])
    return {
        "reactant_graph_batch": Batch.from_data_list(reactant_graphs),
        "product_graph_batch": Batch.from_data_list([item["product_graph"] for item in batch]),
        "target_graph_batch": Batch.from_data_list([item["target_graph"] for item in batch]),
        "fp": torch.stack([item["fp"] for item in batch], dim=0),
        "val": torch.stack([item["val"] for item in batch], dim=0),
        "c_fp": torch.stack([item["c_fp"] for item in batch], dim=0),
        "c_val": torch.stack([item["c_val"] for item in batch], dim=0),
        "rxn_emb": torch.stack([item["rxn_emb"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "target_mask": torch.stack([item["target_mask"] for item in batch], dim=0),
    }
