import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arrange.dataset_specs import get_dataset_spec, resolve_path
from arrange.reaction_io import (
    build_reaction_records,
    load_or_build_rxnfp_json,
    load_rows,
)


@dataclass
class Config:
    # 改这里即可。默认一次性处理当前 arrange 支持的全部数据集。
    datasets: Tuple[str, ...] = ("esol", "freesolve", "lip", "bace", "bbbp", "tox21", "sider")
    output_dir: str = "data/processed"
    sample_size: Optional[int] = None
    reactant_slots: int = 3
    rxnfp_batch_size: int = 64
    force_rebuild: bool = False


def extract_one(dataset_name: str, config: Config) -> Path:
    spec = get_dataset_spec(dataset_name)
    rows = load_rows(spec, sample_size=config.sample_size)
    records = build_reaction_records(spec, rows, reactant_slots=config.reactant_slots)
    reactions = [record["reaction_smiles"] for record in records]
    output_path = resolve_path(config.output_dir) / f"{spec.name}_rxnfp.json"
    embeddings = load_or_build_rxnfp_json(
        reactions,
        json_path=output_path,
        batch_size=config.rxnfp_batch_size,
        force=config.force_rebuild,
        dataset=spec.name,
    )
    print(
        f"{spec.name}: records={len(records)} rxnfp_shape={embeddings.shape} "
        f"saved={output_path}"
    )
    return output_path


def main() -> None:
    config = Config()
    for dataset_name in config.datasets:
        extract_one(dataset_name, config)


if __name__ == "__main__":
    main()
