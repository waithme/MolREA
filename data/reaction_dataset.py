from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..proxy import ProxyEstimator
from .dataset_specs import DatasetSpec
from .reaction_io import graph_or_dummy, mol_fp


class ReactionEffectDataset(Dataset):
    """Dataset for ReactionEffectNet fine-tuning and fusion training."""

    def __init__(
        self,
        spec: DatasetSpec,
        records: Sequence[Dict],
        rxn_embeddings: np.ndarray,
        reactant_slots: int,
        fp_dim: int,
    ) -> None:
        if len(records) != rxn_embeddings.shape[0]:
            raise ValueError("records and rxn_embeddings must have the same length.")
        self.spec = spec
        self.records = list(records)
        self.rxn_embeddings = rxn_embeddings.astype(np.float32, copy=False)
        self.reactant_slots = reactant_slots
        self.fp_dim = fp_dim
        self.value_estimator: Optional[ProxyEstimator] = None
        self._graph_cache: Dict[str, Data] = {}
        self._fp_cache: Dict[str, np.ndarray] = {}
        self._value_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_value_estimator(self, estimator: ProxyEstimator) -> None:
        self.value_estimator = estimator
        self._value_cache.clear()

    def product_smiles_list(self):
        return [record["product_smiles"] for record in self.records]

    def _reactant_value(self, smiles: str) -> np.ndarray:
        if self.value_estimator is None:
            return np.zeros(len(self.spec.metric_names), dtype=np.float32)
        cached = self._value_cache.get(smiles)
        if cached is None:
            cached = self.value_estimator.predict_smiles(smiles).astype(np.float32, copy=False)
            self._value_cache[smiles] = cached
        return cached

    def _graph_for_smiles(self, smiles: str) -> Data:
        cached = self._graph_cache.get(smiles)
        if cached is None:
            cached = graph_or_dummy(smiles)
            self._graph_cache[smiles] = cached
        return cached.clone()

    def _fp_for_smiles(self, smiles: str) -> np.ndarray:
        cached = self._fp_cache.get(smiles)
        if cached is None:
            cached = mol_fp(smiles, self.fp_dim).astype(np.float32, copy=False)
            self._fp_cache[smiles] = cached
        return cached

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        num_tasks = len(self.spec.metric_names)
        reactant_graphs = []
        fp = np.zeros((self.reactant_slots, self.fp_dim), dtype=np.float32)
        val = np.zeros((self.reactant_slots, num_tasks), dtype=np.float32)
        mask = np.zeros(self.reactant_slots, dtype=bool)

        for slot in range(self.reactant_slots):
            if slot < len(record["reactants"]):
                smiles = record["reactants"][slot]
                reactant_graphs.append(self._graph_for_smiles(smiles))
                fp[slot] = self._fp_for_smiles(smiles)
                val[slot] = self._reactant_value(smiles)
                mask[slot] = True
            else:
                reactant_graphs.append(self._graph_for_smiles(""))

        c_val = record["target"].astype(np.float32)

        return {
            "reactant_graphs": reactant_graphs,
            "product_graph": record["product_graph"],
            "fp": torch.from_numpy(fp),
            "val": torch.from_numpy(val),
            "c_fp": torch.from_numpy(self._fp_for_smiles(record["product_smiles"])),
            "c_val": torch.from_numpy(c_val),
            "rxn_emb": torch.from_numpy(self.rxn_embeddings[index]),
            "mask": torch.from_numpy(mask),
            "target": torch.from_numpy(record["target"].astype(np.float32)),
            "target_mask": torch.from_numpy(record["target_mask"].astype(np.float32)),
            "cliff_mol": torch.tensor(float(record.get("cliff_mol", 0)), dtype=torch.float32),
        }
