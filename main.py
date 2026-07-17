import sys
from pathlib import Path

if __package__ in {None, ""}:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    __package__ = package_dir.name

from .pipeline import ArrangePipeline, PipelineConfig


def main() -> None:
    config = PipelineConfig(
        # 只需要在这里改常用参数。
        # datasets=("esol", "freesolve", "lip", "bace", "bbbp", "tox21", "sider"),
        datasets=(
            "chembl2147_ki",
            "chembl244_ki",
            "chembl239_ec50",
            "chembl234_ki",
            "chembl228_ki",
        ),
        seeds=(0,1,2,3,4),
        model_init_seed=0,
        batch_size=128,
        reaction_epochs=50,
        fusion_epochs=100,
        learning_rate=1e-4,
        min_learning_rate=1e-6,
        backbone_learning_rate=1e-5,
        # True/1: keep the current absolute branch behavior.
        # False/0: disable absolute prediction, its loss, and fusion features.
        use_absolute_branch=True,
        device="cuda",
        # auto: MoleculeACE datasets use their official CLIF labels;
        # all other datasets use scaffold splitting.
        split_method="auto",
    )
    ArrangePipeline(config).run()


if __name__ == "__main__":
    main()
