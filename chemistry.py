from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def morgan_count_fp(mol: Optional[Chem.Mol], n_bits: int, radius: int = 2) -> np.ndarray:
    fp = np.zeros(n_bits, dtype=np.float32)
    if mol is None:
        return fp
    sparse_fp = AllChem.GetHashedMorganFingerprint(
        mol,
        radius=radius,
        nBits=n_bits,
        useChirality=True,
    )
    for bit_idx, count in sparse_fp.GetNonzeroElements().items():
        fp[bit_idx] = float(count)
    return fp
