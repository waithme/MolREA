from typing import Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Fragments, rdMolDescriptors
from torch import nn

from .chemistry import mol_from_smiles
from .data.dataset_specs import DatasetSpec
from .data.reaction_io import parse_targets


GENERIC_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "BertzCT",
    "BalabanJ",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "MaxAbsPartialCharge",
    "FormalCharge",
    "AromaticProportion",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SMR_VSA1",
    "SMR_VSA3",
    "EState_VSA1",
    "EState_VSA2",
    "qed",
)


FREESOLVE_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "PEOE_VSA1",
    "PEOE_VSA6",
    "PEOE_VSA10",
    "PEOE_VSA14",
    "SlogP_VSA1",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "SMR_VSA3",
    "SMR_VSA7",
    "EState_VSA1",
    "EState_VSA6",
)


LIP_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "AromaticProportion",
    "PEOE_VSA6",
    "PEOE_VSA12",
    "SlogP_VSA1",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SMR_VSA3",
    "SMR_VSA5",
    "SMR_VSA7",
    "EState_VSA1",
    "EState_VSA6",
)


BACE_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHeterocycles",
    "NumAromaticHeterocycles",
    "NumSaturatedRings",
    "AromaticProportion",
    "PEOE_VSA6",
    "PEOE_VSA12",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "SMR_VSA3",
    "EState_VSA6",
    "fr_amide",
    "fr_NH0",
)


BBBP_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumAromaticHeterocycles",
    "NumSaturatedRings",
    "AromaticProportion",
    "PEOE_VSA6",
    "PEOE_VSA12",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "EState_VSA6",
    "fr_NH0",
    "fr_Ar_N",
)


TOX21_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumAromaticHeterocycles",
    "NumHeterocycles",
    "NumSaturatedRings",
    "AromaticProportion",
    "HalogenFraction",
    "PEOE_VSA3",
    "PEOE_VSA6",
    "PEOE_VSA10",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "EState_VSA6",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "fr_Ar_N",
    "fr_aniline",
    "fr_NH0",
    "qed",
)


SIDER_DESCRIPTOR_NAMES = (
    "fr_allylic_oxid",
    "RingCount",
    "AvgIpc",
    "MaxEStateIndex",
    "MaxAbsEStateIndex",
    "NumAliphaticCarbocycles",
    "PEOE_VSA3",
    "fr_sulfide",
    "SlogP_VSA10",
    "NumAromaticHeterocycles",
    "qed",
    "fr_ketone",
    "fr_aniline",
    "NumHeterocycles",
    "FpDensityMorgan3",
    "fr_lactam",
    "NumAliphaticRings",
    "BalabanJ",
    "PEOE_VSA13",
    "NumAromaticRings",
    "fr_Ar_N",
    "NumAromaticCarbocycles",
    "fr_benzene",
    "NumSaturatedCarbocycles",
    "fr_NH0",
    "fr_term_acetylene",
    "BCUT2D_LOGPHI",
    "EState_VSA9",
    "VSA_EState10",
    "fr_bicyclic",
)


CHEMBL228_KI_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAromaticHeterocycles",
    "AromaticProportion",
    "HalogenFraction",
    "PEOE_VSA6",
    "PEOE_VSA12",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "SMR_VSA7",
    "EState_VSA6",
    "FpDensityMorgan2",
    "fr_Ar_N",
    "fr_aniline",
    "fr_NH0",
    "fr_NH1",
    "qed",
)


CHEMBL234_KI_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAromaticHeterocycles",
    "NumSaturatedRings",
    "AromaticProportion",
    "HalogenFraction",
    "PEOE_VSA6",
    "PEOE_VSA12",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "SMR_VSA7",
    "EState_VSA6",
    "FpDensityMorgan2",
    "fr_Ar_N",
    "fr_NH0",
    "fr_NH1",
    "qed",
)


CHEMBL239_EC50_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "AromaticProportion",
    "HalogenFraction",
    "PEOE_VSA1",
    "PEOE_VSA6",
    "PEOE_VSA10",
    "SlogP_VSA1",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "SMR_VSA3",
    "EState_VSA1",
    "EState_VSA6",
    "fr_COO",
    "fr_COO2",
    "qed",
)


CHEMBL244_KI_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAromaticHeterocycles",
    "AromaticProportion",
    "HalogenFraction",
    "PEOE_VSA6",
    "PEOE_VSA12",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "SMR_VSA3",
    "EState_VSA6",
    "FpDensityMorgan2",
    "fr_amide",
    "fr_amidine",
    "fr_guanido",
    "fr_NH0",
    "qed",
)


CHEMBL2147_KI_DESCRIPTOR_NAMES = (
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "FormalCharge",
    "MaxAbsPartialCharge",
    "MolWt",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "FractionCSP3",
    "LabuteASA",
    "MolMR",
    "RingCount",
    "NumAromaticRings",
    "NumAromaticHeterocycles",
    "NumHeterocycles",
    "AromaticProportion",
    "HalogenFraction",
    "PEOE_VSA6",
    "PEOE_VSA12",
    "SlogP_VSA4",
    "SlogP_VSA10",
    "SMR_VSA7",
    "EState_VSA6",
    "FpDensityMorgan2",
    "fr_Ar_N",
    "fr_aniline",
    "fr_NH0",
    "qed",
)


CHEMBL_DESCRIPTOR_NAMES_BY_KIND = {
    "chembl228_ki": CHEMBL228_KI_DESCRIPTOR_NAMES,
    "chembl234_ki": CHEMBL234_KI_DESCRIPTOR_NAMES,
    "chembl239_ec50": CHEMBL239_EC50_DESCRIPTOR_NAMES,
    "chembl244_ki": CHEMBL244_KI_DESCRIPTOR_NAMES,
    "chembl2147_ki": CHEMBL2147_KI_DESCRIPTOR_NAMES,
}


def _descriptor_functions():
    functions = {name: fn for name, fn in Descriptors.descList}
    for name in dir(Fragments):
        if name.startswith("fr_") and name not in functions:
            functions[name] = getattr(Fragments, name)
    return functions


DESCRIPTOR_FUNCTIONS = _descriptor_functions()


class ProxyEstimator(Protocol):
    def fit(self, rows: Sequence[dict]) -> None:
        ...

    def predict_smiles(self, smiles: str) -> np.ndarray:
        ...


def max_abs_partial_charge(mol: Chem.Mol) -> float:
    try:
        mol_h = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol_h)
        charges = []
        for atom in mol_h.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                try:
                    value = abs(float(atom.GetProp("_GasteigerCharge")))
                except ValueError:
                    continue
                if np.isfinite(value):
                    charges.append(value)
        return max(charges) if charges else 0.0
    except Exception:
        return 0.0


def halogen_fraction(mol: Chem.Mol) -> float:
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms <= 0:
        return 0.0
    halogen_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53})
    return halogen_atoms / heavy_atoms


def generic_descriptors(smiles: str) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    values = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.MolMR(mol),
        Descriptors.BertzCT(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.Kappa3(mol),
        max_abs_partial_charge(mol),
        Chem.GetFormalCharge(mol),
        aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0,
        Descriptors.SlogP_VSA1(mol),
        Descriptors.SlogP_VSA2(mol),
        Descriptors.SlogP_VSA3(mol),
        Descriptors.SMR_VSA1(mol),
        Descriptors.SMR_VSA3(mol),
        Descriptors.EState_VSA1(mol),
        Descriptors.EState_VSA2(mol),
        Descriptors.qed(mol),
    ]
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def freesolve_descriptors(smiles: str) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    # Hydration free energy is dominated by polarity, charge distribution,
    # hydrogen-bonding, and how those properties are exposed over molecular surface.
    values = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHeteroatoms(mol),
        float(Chem.GetFormalCharge(mol)),
        max_abs_partial_charge(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.MolMR(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        Descriptors.PEOE_VSA1(mol),
        Descriptors.PEOE_VSA6(mol),
        Descriptors.PEOE_VSA10(mol),
        Descriptors.PEOE_VSA14(mol),
        Descriptors.SlogP_VSA1(mol),
        Descriptors.SlogP_VSA4(mol),
        Descriptors.SlogP_VSA10(mol),
        Descriptors.SMR_VSA3(mol),
        Descriptors.SMR_VSA7(mol),
        Descriptors.EState_VSA1(mol),
        Descriptors.EState_VSA6(mol),
    ]
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def lip_descriptors(smiles: str) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    # Lipophilicity is driven primarily by hydrophobic surface, polarity,
    # charge state, hydrogen-bonding capacity, and polarizability.
    values = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHeteroatoms(mol),
        float(Chem.GetFormalCharge(mol)),
        max_abs_partial_charge(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.MolMR(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0,
        Descriptors.PEOE_VSA6(mol),
        Descriptors.PEOE_VSA12(mol),
        Descriptors.SlogP_VSA1(mol),
        Descriptors.SlogP_VSA4(mol),
        Descriptors.SlogP_VSA5(mol),
        Descriptors.SlogP_VSA10(mol),
        Descriptors.SlogP_VSA11(mol),
        Descriptors.SMR_VSA3(mol),
        Descriptors.SMR_VSA5(mol),
        Descriptors.SMR_VSA7(mol),
        Descriptors.EState_VSA1(mol),
        Descriptors.EState_VSA6(mol),
    ]
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def bace_descriptors(smiles: str) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    values = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHeteroatoms(mol),
        Chem.GetFormalCharge(mol),
        max_abs_partial_charge(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.MolMR(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0,
        Descriptors.PEOE_VSA6(mol),
        Descriptors.PEOE_VSA12(mol),
        Descriptors.SlogP_VSA4(mol),
        Descriptors.SlogP_VSA10(mol),
        Descriptors.SMR_VSA3(mol),
        Descriptors.EState_VSA6(mol),
        Fragments.fr_amide(mol),
        Fragments.fr_NH0(mol),
    ]
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def bbbp_descriptors(smiles: str) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    values = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHeteroatoms(mol),
        Chem.GetFormalCharge(mol),
        max_abs_partial_charge(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.MolMR(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0,
        Descriptors.PEOE_VSA6(mol),
        Descriptors.PEOE_VSA12(mol),
        Descriptors.SlogP_VSA4(mol),
        Descriptors.SlogP_VSA10(mol),
        Descriptors.EState_VSA6(mol),
        Fragments.fr_NH0(mol),
        Fragments.fr_Ar_N(mol),
    ]
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def tox21_descriptors(smiles: str) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    values = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHeteroatoms(mol),
        Chem.GetFormalCharge(mol),
        max_abs_partial_charge(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0,
        halogen_fraction(mol),
        Descriptors.PEOE_VSA3(mol),
        Descriptors.PEOE_VSA6(mol),
        Descriptors.PEOE_VSA10(mol),
        Descriptors.SlogP_VSA4(mol),
        Descriptors.SlogP_VSA10(mol),
        Descriptors.EState_VSA6(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Fragments.fr_Ar_N(mol),
        Fragments.fr_aniline(mol),
        Fragments.fr_NH0(mol),
        Descriptors.qed(mol),
    ]
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def selected_descriptors(smiles: str, names: Sequence[str]) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    custom_values = {
        "FormalCharge": float(Chem.GetFormalCharge(mol)),
        "MaxAbsPartialCharge": max_abs_partial_charge(mol),
        "AromaticProportion": aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0,
        "HalogenFraction": halogen_fraction(mol),
    }
    values = []
    for name in names:
        if name in custom_values:
            values.append(custom_values[name])
            continue
        fn = DESCRIPTOR_FUNCTIONS.get(name)
        if fn is None:
            values.append(0.0)
            continue
        try:
            values.append(float(fn(mol)))
        except Exception:
            values.append(0.0)
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def descriptor_features(smiles: str, spec: DatasetSpec) -> Optional[np.ndarray]:
    if spec.proxy_kind in CHEMBL_DESCRIPTOR_NAMES_BY_KIND:
        return selected_descriptors(smiles, CHEMBL_DESCRIPTOR_NAMES_BY_KIND[spec.proxy_kind])
    if spec.proxy_kind == "sider":
        return selected_descriptors(smiles, SIDER_DESCRIPTOR_NAMES)
    if spec.proxy_kind == "freesolve":
        return freesolve_descriptors(smiles)
    if spec.proxy_kind == "lip":
        return lip_descriptors(smiles)
    if spec.proxy_kind == "bace":
        return bace_descriptors(smiles)
    if spec.proxy_kind == "bbbp":
        return bbbp_descriptors(smiles)
    if spec.proxy_kind == "tox21":
        return tox21_descriptors(smiles)
    return generic_descriptors(smiles)


def descriptor_dim(spec: DatasetSpec) -> int:
    if spec.proxy_kind in CHEMBL_DESCRIPTOR_NAMES_BY_KIND:
        return len(CHEMBL_DESCRIPTOR_NAMES_BY_KIND[spec.proxy_kind])
    if spec.proxy_kind == "sider":
        return len(SIDER_DESCRIPTOR_NAMES)
    if spec.proxy_kind == "freesolve":
        return len(FREESOLVE_DESCRIPTOR_NAMES)
    if spec.proxy_kind == "lip":
        return len(LIP_DESCRIPTOR_NAMES)
    if spec.proxy_kind == "bace":
        return len(BACE_DESCRIPTOR_NAMES)
    if spec.proxy_kind == "bbbp":
        return len(BBBP_DESCRIPTOR_NAMES)
    if spec.proxy_kind == "tox21":
        return len(TOX21_DESCRIPTOR_NAMES)
    return len(GENERIC_DESCRIPTOR_NAMES)


def _rows_to_training_arrays(spec: DatasetSpec, rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    masks = []
    for row in rows:
        descriptor = descriptor_features(row[spec.smiles_column], spec)
        if descriptor is None:
            continue
        y, y_mask = parse_targets(row, spec)
        if y_mask.sum() <= 0:
            continue
        features.append(descriptor)
        targets.append(y)
        masks.append(y_mask)
    if not features:
        raise ValueError(f"No valid rows to fit proxy for {spec.name}.")
    x = np.stack(features).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    mask = np.stack(masks).astype(bool)
    return x, y, mask


def create_proxy_estimator(
    spec: DatasetSpec,
    seed: int,
    model_init_seed: int = 0,
    epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> ProxyEstimator:
    return DescriptorProxy(
        spec,
        seed=seed,
        model_init_seed=model_init_seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


class DescriptorProxy:
    def __init__(
        self,
        spec: DatasetSpec,
        seed: int,
        model_init_seed: int = 0,
        hidden_layers: Tuple[int, ...] = (128, 64),
        epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        val_ratio: float = 0.1,
        early_stopping_patience: int = 30,
    ) -> None:
        self.spec = spec
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_init_seed = model_init_seed
        self.dropout = dropout
        self.val_ratio = val_ratio
        self.early_stopping_patience = early_stopping_patience
        self.num_tasks = len(spec.target_columns)
        self.is_binary = spec.task_type in {"binary", "multitask_binary"}
        sample_dim = descriptor_dim(spec)
        layers = []
        in_dim = sample_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_tasks))
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.model_init_seed)
            self.model = nn.Sequential(*layers)
        finally:
            torch.random.set_rng_state(rng_state)
        self.x_mean = torch.zeros(sample_dim)
        self.x_std = torch.ones(sample_dim)
        self.y_mean = torch.zeros(self.num_tasks)
        self.y_std = torch.ones(self.num_tasks)
        self.fitted = False

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.is_binary:
            loss_all = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            loss_all = (pred - target) ** 2
        return loss_all.masked_select(mask).mean() if bool(mask.any()) else loss_all.sum() * 0.0

    def fit(self, rows: Sequence[dict]) -> None:
        x_np, y_np, mask_np = _rows_to_training_arrays(self.spec, rows)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        mask = torch.from_numpy(mask_np)
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        x = (x - self.x_mean) / self.x_std
        if not self.is_binary:
            valid_values = y[mask]
            self.y_mean = valid_values.mean().view(1).repeat(self.num_tasks)
            self.y_std = valid_values.std(unbiased=False).clamp_min(1e-6).view(1).repeat(self.num_tasks)
            y_train = (y - self.y_mean) / self.y_std
        else:
            y_train = y

        generator = torch.Generator().manual_seed(self.seed)
        num_samples = x.size(0)
        permutation = torch.randperm(num_samples, generator=generator)
        val_size = int(num_samples * self.val_ratio) if num_samples >= 20 else 0
        if val_size > 0:
            val_idx = permutation[:val_size]
            train_idx = permutation[val_size:]
        else:
            train_idx = permutation
            val_idx = torch.empty(0, dtype=torch.long)

        x_train, y_train_split, mask_train = x[train_idx], y_train[train_idx], mask[train_idx]
        x_val, y_val_split, mask_val = x[val_idx], y_train[val_idx], mask[val_idx]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
        )
        self.model.train()
        best_state = None
        best_val_loss = float("inf")
        patience_left = self.early_stopping_patience
        train_size = x_train.size(0)
        for _ in range(self.epochs):
            epoch_perm = torch.randperm(train_size, generator=generator)
            for start in range(0, train_size, self.batch_size):
                idx = epoch_perm[start:start + self.batch_size]
                pred = self.model(x_train[idx])
                loss = self._masked_loss(pred, y_train_split[idx], mask_train[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if val_idx.numel() == 0:
                continue

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(x_val)
                val_loss = float(self._masked_loss(val_pred, y_val_split, mask_val).item())
            self.model.train()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience_left = self.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.fitted = True

    def predict_smiles(self, smiles: str) -> np.ndarray:
        descriptor = descriptor_features(smiles, self.spec)
        if descriptor is None or not self.fitted:
            return np.zeros(self.num_tasks, dtype=np.float32)
        x = torch.from_numpy(descriptor).view(1, -1).float()
        x = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            pred = self.model(x).view(-1)
            if self.is_binary:
                pred = torch.sigmoid(pred)
            else:
                pred = pred * self.y_std + self.y_mean
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred.detach().cpu().numpy().astype(np.float32)
