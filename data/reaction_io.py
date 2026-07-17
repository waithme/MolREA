import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch_geometric.data import Data

from ..chemistry import (
    mol_from_smiles,
    morgan_count_fp,
)
from ..models.reaction_effect_net_gin_multitask import dummy_graph, mol_to_graph
from .dataset_specs import DatasetSpec, resolve_path


def load_rows(spec: DatasetSpec, sample_size: Optional[int] = None) -> List[Dict[str, str]]:
    path = resolve_path(spec.data_path)
    if path.suffix.lower() == ".json":
        return load_rows_from_json(spec, path, sample_size)

    import csv

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {spec.id_column, spec.smiles_column, *spec.target_columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
            if sample_size is not None and len(rows) >= sample_size:
                break
    return rows


def load_rows_from_json(spec: DatasetSpec, path: Path, sample_size: Optional[int] = None) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object keyed by {spec.id_column}.")

    rows: List[Dict[str, str]] = []
    for key, item in payload.items():
        if not isinstance(item, dict):
            raise ValueError(f"{path} entry '{key}' must be an object.")
        smiles = item.get("smiles", "")
        labels = item.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"{path} entry '{key}' must contain a list field 'labels'.")

        row = {
            spec.id_column: str(key),
            spec.smiles_column: str(smiles),
        }
        for idx, column in enumerate(spec.target_columns):
            value = labels[idx] if idx < len(labels) else ""
            row[column] = "" if value is None else str(value)
        rows.append(row)
        if sample_size is not None and len(rows) >= sample_size:
            break
    return rows


def parse_targets(row: Dict[str, str], spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    mask = []
    for column in spec.target_columns:
        raw = row.get(column, "")
        if raw is None or str(raw).strip() == "":
            values.append(0.0)
            mask.append(0.0)
            continue
        try:
            value = float(raw)
            if not np.isfinite(value):
                values.append(0.0)
                mask.append(0.0)
                continue
            values.append(value)
            mask.append(1.0)
        except ValueError:
            values.append(0.0)
            mask.append(0.0)
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def load_reaction_json(path: str) -> Dict:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reaction_rank(metadata: Dict) -> int:
    rank = metadata.get("expansion_rank")
    return rank if isinstance(rank, int) else 10**9


def select_reactants_from_reaction_json(entries) -> List[str]:
    if not isinstance(entries, list):
        return []
    candidates = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        reactants, metadata = entry[0], entry[1]
        if not isinstance(reactants, list) or not isinstance(metadata, dict):
            continue
        smiles = [item for item in reactants if isinstance(item, str) and item]
        if smiles:
            candidates.append((reaction_rank(metadata), smiles))
    if not candidates:
        return []
    return min(candidates, key=lambda item: item[0])[1]


def make_forward_reaction_smiles(reactants: Sequence[str], product_smiles: str) -> Optional[str]:
    reactants = [smiles for smiles in reactants if smiles]
    if not reactants or not product_smiles:
        return None
    return f"{'.'.join(reactants)}>>{product_smiles}"


def valid_product_graph(smiles: str) -> Optional[Data]:
    if mol_from_smiles(smiles) is None:
        return None
    try:
        return mol_to_graph(smiles)
    except Exception:
        return None


def mol_fp(smiles: str, fp_dim: int) -> np.ndarray:
    return morgan_count_fp(mol_from_smiles(smiles), n_bits=fp_dim)


def build_reaction_records(
    spec: DatasetSpec,
    rows: Sequence[Dict[str, str]],
    reactant_slots: int,
) -> List[Dict]:
    reaction_data = load_reaction_json(spec.reaction_json_path)
    records = []
    for row in rows:
        product_smiles = row.get(spec.smiles_column, "")
        key = row.get(spec.id_column, "")
        y, y_mask = parse_targets(row, spec)
        if y_mask.sum() <= 0:
            continue
        reactants = select_reactants_from_reaction_json(reaction_data.get(key))
        reaction_smiles = make_forward_reaction_smiles(reactants, product_smiles)
        graph = valid_product_graph(product_smiles)
        has_reaction_context = graph is not None and reaction_smiles is not None
        records.append(
            {
                "id": key,
                "product_smiles": product_smiles,
                "reactants": reactants[:reactant_slots] if has_reaction_context else [],
                "reaction_smiles": reaction_smiles,
                "has_reaction_context": has_reaction_context,
                "target": y,
                "target_mask": y_mask,
                "product_graph": graph if graph is not None else dummy_graph(),
                "split": str(row.get("split", "")).strip().lower(),
                "cliff_mol": int(float(row.get("cliff_mol", 0) or 0)),
            }
        )
    return records


def load_rxn_embeddings(records: Sequence[Dict], rxnfp_json_path: str) -> np.ndarray:
    path = resolve_path(rxnfp_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing precomputed RXNFP JSON: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object keyed by record id.")

    embedding_dim = next(
        (
            len(value)
            for value in payload.values()
            if isinstance(value, list) and value
        ),
        None,
    )
    if embedding_dim is None:
        raise ValueError(f"{path} does not contain any valid RXNFP embeddings.")

    embeddings: List[List[float]] = []
    for record in records:
        record_id = str(record["id"])
        embedding = payload.get(record_id)
        if (
            not record.get("has_reaction_context", False)
            or not isinstance(embedding, list)
            or len(embedding) != embedding_dim
        ):
            embeddings.append([0.0] * embedding_dim)
        else:
            embeddings.append(embedding)

    rxn_embeddings = np.asarray(embeddings, dtype=np.float32)
    if rxn_embeddings.ndim != 2:
        raise ValueError(f"{path} must contain 2D RXNFP embeddings, got shape {rxn_embeddings.shape}")
    return rxn_embeddings


def graph_or_dummy(smiles: str) -> Data:
    try:
        return mol_to_graph(smiles)
    except Exception:
        return dummy_graph()
