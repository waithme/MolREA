from typing import List, Sequence

from rdkit import Chem


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_DIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def safe_index(values: Sequence, value) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values)


def atom_features(atom: Chem.rdchem.Atom) -> List[int]:
    return [
        safe_index(ATOM_LIST, atom.GetAtomicNum()),
        safe_index(CHIRALITY_LIST, atom.GetChiralTag()),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        safe_index(HYBRIDIZATION_LIST, atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]


def bond_features(bond: Chem.rdchem.Bond) -> List[int]:
    return [
        safe_index(BOND_TYPE_LIST, bond.GetBondType()),
        safe_index(BOND_DIR_LIST, bond.GetBondDir()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]
