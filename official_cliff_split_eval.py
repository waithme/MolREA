import sys
from pathlib import Path

if __package__ in {None, ""}:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    __package__ = package_dir.name

from .pipeline import ArrangePipeline, PipelineConfig


CLIFF_DATASETS = (
    "chembl244_ki",
    "chembl239_ec50",
    "chembl234_ki",
    "chembl228_ki",
    "chembl2147_ki",
)


def main() -> None:
    """Run the official MoleculeACE CLIF splits through the unified pipeline."""
    config = PipelineConfig(
        datasets=CLIFF_DATASETS,
        seeds=(0, 1, 2, 3, 4),
        model_init_seed=0,
        split_method="auto",
        batch_size=128,
        reaction_epochs=50,
        fusion_epochs=100,
        learning_rate=1e-4,
        min_learning_rate=1e-6,
        backbone_learning_rate=1e-5,
        results_csv="results/official_cliff_split/summary.csv",
    )
    ArrangePipeline(config).run()


if __name__ == "__main__":
    main()
