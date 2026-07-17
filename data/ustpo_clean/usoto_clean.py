import os
import re
import argparse
from typing import List, Optional, Tuple, Dict

import pandas as pd
from tqdm import tqdm
from rdkit import Chem


def extract_first_reaction_smiles(line: str) -> Optional[str]:
    """
    USPTO txt files may contain reaction SMILES plus extra fields.
    Usually the first whitespace-separated field is the reaction SMILES.
    """
    line = line.strip()
    if not line:
        return None

    # Take the first column/token
    rxn = line.split()[0].strip()

    if not rxn:
        return None

    return rxn


def split_reaction_smiles(rxn: str) -> Optional[Tuple[str, str, str]]:
    """
    Support both:
        reactants>reagents>products
    and:
        reactants>>products

    Return:
        reactants, reagents, products
    """
    if not rxn or ">" not in rxn:
        return None

    parts = rxn.split(">")

    if len(parts) == 3:
        reactants, reagents, products = parts
        return reactants.strip(), reagents.strip(), products.strip()

    # Extremely defensive fallback
    if ">>" in rxn:
        reactants, products = rxn.split(">>")
        return reactants.strip(), "", products.strip()

    return None


def split_molecules(smiles_block: str) -> List[str]:
    """
    Split molecules separated by '.'.
    Remove empty strings.
    """
    if smiles_block is None or smiles_block == "":
        return []

    return [s.strip() for s in smiles_block.split(".") if s.strip()]


def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def is_valid_mol_smiles(smi: str) -> bool:
    return mol_from_smiles(smi) is not None


def has_atom_mapping(smi: str) -> bool:
    """
    Check whether a molecule/reaction SMILES contains atom mapping like [C:12].
    """
    return re.search(r":\d+\]", smi) is not None


def remove_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
    """
    Return a copy of mol with atom map numbers removed.
    """
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonicalize_smiles(
    smi: str,
    remove_mapping: bool = False,
    isomeric: bool = True,
) -> Optional[str]:
    """
    Canonicalize a molecule SMILES.
    Optionally remove atom mapping.
    """
    mol = mol_from_smiles(smi)
    if mol is None:
        return None

    if remove_mapping:
        mol = remove_atom_mapping(mol)

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def canonicalize_molecule_list(
    smiles_list: List[str],
    remove_mapping: bool = False,
    sort_mols: bool = True,
) -> Optional[List[str]]:
    """
    Canonicalize a list of molecule SMILES.
    """
    out = []

    for smi in smiles_list:
        can = canonicalize_smiles(smi, remove_mapping=remove_mapping)
        if can is None:
            return None
        out.append(can)

    if sort_mols:
        out = sorted(out)

    return out


def process_one_reaction(
    rxn: str,
    max_reactants: int = 3,
    require_atom_mapping: bool = True,
    require_single_product: bool = True,
) -> Optional[str]:
    """
    Convert one USPTO reaction to:
        reactants>>product

    Returns the unmapped canonical reaction SMILES if valid, otherwise None.
    """
    split = split_reaction_smiles(rxn)
    if split is None:
        return None

    reactants_block, reagents_block, products_block = split

    reactants = split_molecules(reactants_block)
    products = split_molecules(products_block)

    # Basic checks
    if len(reactants) == 0 or len(products) == 0:
        return None

    if len(reactants) > max_reactants:
        return None

    if require_single_product and len(products) != 1:
        return None

    # Remove reagents/agents by design.
    # Check atom mapping on the original mapped reaction before removing maps.
    mapped_reaction = f"{'.'.join(reactants)}>{'.'.join(products)}"

    if require_atom_mapping and not has_atom_mapping(mapped_reaction):
        return None

    # RDKit parse check
    for smi in reactants + products:
        if not is_valid_mol_smiles(smi):
            return None

    # Canonical unmapped SMILES
    # This is the only representation needed by ReactionEffectNet-style loaders.
    reactants_unmapped_can = canonicalize_molecule_list(
        reactants,
        remove_mapping=True,
        sort_mols=True,
    )
    products_unmapped_can = canonicalize_molecule_list(
        products,
        remove_mapping=True,
        sort_mols=True,
    )

    if reactants_unmapped_can is None or products_unmapped_can is None:
        return None

    return f"{'.'.join(reactants_unmapped_can)}>>{'.'.join(products_unmapped_can)}"


def find_split_files(data_dir: str) -> Dict[str, str]:
    """
    Find train/valid/test txt files.
    Also tolerate common typo-like names.
    """
    candidates = {
        "train": ["train.txt", "traub.txt", "trn.txt"],
        "valid": ["valid.txt", "val.txt", "dev.txt"],
        "test": ["test.txt"],
    }

    found = {}

    for split, names in candidates.items():
        for name in names:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                found[split] = path
                break

    return found


def process_file(
    path: str,
    split_name: str,
    max_reactants: int = 3,
    require_atom_mapping: bool = True,
    require_single_product: bool = True,
) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Processing {split_name}"):
        rxn = extract_first_reaction_smiles(line)
        if rxn is None:
            continue

        reaction = process_one_reaction(
            rxn,
            max_reactants=max_reactants,
            require_atom_mapping=require_atom_mapping,
            require_single_product=require_single_product,
        )

        if reaction is None:
            continue

        rows.append({"reactions": reaction})

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.txt, valid.txt, and test.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_uspto",
        help="Output directory",
    )
    parser.add_argument(
        "--max_reactants",
        type=int,
        default=3,
        help="Keep reactions with number of reactants <= max_reactants",
    )
    parser.add_argument(
        "--no_require_atom_mapping",
        action="store_true",
        help="If set, do not require atom mapping",
    )
    parser.add_argument(
        "--allow_multi_product",
        action="store_true",
        help="If set, allow multiple products",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate by canonical reaction SMILES",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split_files = find_split_files(args.data_dir)

    if not split_files:
        raise FileNotFoundError(
            f"No train/valid/test txt files found in {args.data_dir}"
        )

    print("Found files:")
    for split, path in split_files.items():
        print(f"  {split}: {path}")

    all_dfs = []

    for split, path in split_files.items():
        df = process_file(
            path=path,
            split_name=split,
            max_reactants=args.max_reactants,
            require_atom_mapping=not args.no_require_atom_mapping,
            require_single_product=not args.allow_multi_product,
        )

        print(f"{split}: kept {len(df)} reactions")

        split_out = os.path.join(args.output_dir, f"uspto_{split}_filtered.csv")
        df.to_csv(split_out, index=False)
        print(f"Saved {split_out}")

        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)

    print("Before dedup:", len(all_df))

    if args.deduplicate:
        all_df = all_df.drop_duplicates(subset=["reactions"]).reset_index(drop=True)
        print("After dedup by reactions:", len(all_df))

    combined_out = os.path.join(args.output_dir, "uspto.csv")
    all_df.to_csv(combined_out, index=False)
    print(f"Saved combined file: {combined_out}")

    print("\nFinal statistics:")
    print(f"Total reactions: {len(all_df)}")


if __name__ == "__main__":
    main()
