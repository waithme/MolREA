"""Precompute RXNFP vectors for a registered dataset."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ in {None, ""}:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent.parent))
    __package__ = f"{package_dir.parent.name}.{package_dir.name}"

from ..data.dataset_specs import DATASET_SPECS, get_dataset_spec, resolve_path
from ..data.reaction_io import (
    load_reaction_json,
    load_rows,
    make_forward_reaction_smiles,
    select_reactants_from_reaction_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RXNFP embeddings from precomputed reaction routes."
    )
    parser.add_argument("dataset", choices=sorted(DATASET_SPECS))
    parser.add_argument("--reaction-json", help="Override the reaction JSON path.")
    parser.add_argument("--output", help="Override the DatasetSpec RXNFP JSON path.")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def build_reactions(dataset: str, reaction_json_path: str) -> tuple[List[str], List[str]]:
    spec = get_dataset_spec(dataset)
    reaction_data = load_reaction_json(reaction_json_path)
    keys: List[str] = []
    reactions: List[str] = []

    for row in load_rows(spec):
        molecule_id = str(row.get(spec.id_column, "")).strip()
        product_smiles = str(row.get(spec.smiles_column, "")).strip()
        reactants = select_reactants_from_reaction_json(
            reaction_data.get(molecule_id)
        )
        reaction_smiles = make_forward_reaction_smiles(reactants, product_smiles)
        if molecule_id and reaction_smiles:
            keys.append(molecule_id)
            reactions.append(reaction_smiles)

    return keys, reactions


def generate_embeddings(
    reactions: Sequence[str],
    batch_size: int,
) -> List[List[float]]:
    try:
        from rxnfp.models import get_default_model_and_tokenizer
        from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError(
            "The rxnfp package is required. Install it before running this command."
        ) from exc

    model, tokenizer = get_default_model_and_tokenizer()
    generator = RXNBERTFingerprintGenerator(model, tokenizer)
    vectors: List[List[float]] = []
    for start in range(0, len(reactions), batch_size):
        batch = reactions[start:start + batch_size]
        vectors.extend(
            [list(map(float, vector)) for vector in generator.convert_batch(batch)]
        )
        print(f"Embedded {min(start + batch_size, len(reactions))}/{len(reactions)}")
    return vectors


def save_embeddings(
    output_path: Path,
    keys: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> None:
    if len(keys) != len(vectors):
        raise ValueError(
            f"Embedding count mismatch: {len(keys)} keys and {len(vectors)} vectors."
        )
    payload: Dict[str, Sequence[float]] = dict(zip(keys, vectors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    dimension = len(vectors[0]) if vectors else 0
    print(f"Saved {len(payload)} embeddings (dim={dimension}) to {output_path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    spec = get_dataset_spec(args.dataset)
    reaction_json_path = args.reaction_json or spec.reaction_json_path
    output_path = resolve_path(args.output or spec.rxnfp_json_path)
    keys, reactions = build_reactions(args.dataset, reaction_json_path)
    if not reactions:
        raise ValueError(f"No valid reaction contexts found for {args.dataset}.")

    print(f"Selected {len(reactions)} reactions for {args.dataset}.")
    vectors = generate_embeddings(reactions, args.batch_size)
    save_embeddings(output_path, keys, vectors)


if __name__ == "__main__":
    main()
