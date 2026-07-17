"""Generate one-step retrosynthesis candidates for a registered dataset."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

if __package__ in {None, ""}:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent.parent))
    __package__ = f"{package_dir.parent.name}.{package_dir.name}"

from ..data.dataset_specs import DATASET_SPECS, get_dataset_spec, resolve_path
from ..data.reaction_io import load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AiZynthFinder one-step retrosynthesis candidates."
    )
    parser.add_argument("dataset", choices=sorted(DATASET_SPECS))
    parser.add_argument("--config", required=True, help="AiZynthFinder config YAML.")
    parser.add_argument("--policy", default="uspto")
    parser.add_argument("--output", help="Override the DatasetSpec reaction JSON path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output instead of resuming missing molecule IDs.",
    )
    return parser.parse_args()


def molecule_inputs(dataset: str) -> Iterable[Tuple[str, str]]:
    spec = get_dataset_spec(dataset)
    for row in load_rows(spec):
        molecule_id = str(row.get(spec.id_column, "")).strip()
        smiles = str(row.get(spec.smiles_column, "")).strip()
        if molecule_id and smiles:
            yield molecule_id, smiles


def serialize_expansion(expansion) -> list:
    reaction = expansion[0]
    reactants = [molecule.smiles for molecule in reaction.reactants[0]]
    return [reactants, reaction.metadata]


def load_existing(path: Path, overwrite: bool) -> Dict[str, list]:
    if overwrite or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def save_json(path: Path, payload: Dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def generate_reactions(
    dataset: str,
    config_path: Path,
    policy: str,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    try:
        from aizynthfinder.aizynthfinder import AiZynthExpander
    except ImportError as exc:
        raise RuntimeError(
            "AiZynthFinder is required. Install it before running this command."
        ) from exc

    rows = list(molecule_inputs(dataset))
    results = load_existing(output_path, overwrite)
    pending = [(key, smiles) for key, smiles in rows if key not in results]

    print(f"Loading AiZynthFinder config: {config_path}")
    expander = AiZynthExpander(configfile=str(config_path))
    expander.expansion_policy.select(policy)
    expander.filter_policy.select(policy)

    failures = 0
    for index, (molecule_id, smiles) in enumerate(pending, start=1):
        try:
            results[molecule_id] = [
                serialize_expansion(expansion)
                for expansion in expander.do_expansion(smiles)
            ]
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(pending)}] {molecule_id}: failed: {exc}")
        else:
            print(f"[{index}/{len(pending)}] {molecule_id}: done")

        if index % 25 == 0:
            save_json(output_path, results)

    save_json(output_path, results)
    print(
        f"Saved {len(results)} molecules to {output_path}; "
        f"processed={len(pending)}, failed={failures}."
    )


def main() -> None:
    args = parse_args()
    spec = get_dataset_spec(args.dataset)
    output_path = resolve_path(args.output or spec.reaction_json_path)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"AiZynthFinder config not found: {config_path}")
    generate_reactions(
        dataset=args.dataset,
        config_path=config_path,
        policy=args.policy,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
