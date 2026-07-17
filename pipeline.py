import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .data.dataset_specs import get_dataset_spec, resolve_path
from .fusion import FusionConfig, train_fusion
from .reaction_finetune import ReactionFinetuneConfig, build_dataset, train_reaction_effect


@dataclass
class PipelineConfig:
    datasets: Tuple[str, ...] = ("esol", "freesolve", "lip", "bace", "bbbp", "tox21", "sider")
    seeds: Tuple[int, ...] = (0,)
    dataset_seed_overrides: Dict[str, Tuple[int, ...]] = None
    model_init_seed: int = 0
    split_method: str = "auto"
    batch_size: int = 128
    reaction_epochs: int = 50
    fusion_epochs: int = 100
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    backbone_learning_rate: float = 1e-5
    device: str = "cuda"
    sample_size: Optional[int] = None
    run_reaction_finetune: bool = True
    run_fusion: bool = True
    use_absolute_branch: bool = True
    num_workers: int = 0
    val_ratio: float = 0.09
    test_ratio: float = 0.1
    pretrained_backbone_path: str = "pretrain_backbone/best_backbone.pt"
    results_csv: str = "results/summary.csv"


def dataset_prefix(dataset_name: str, seed: int) -> str:
    return f"{dataset_name}/seed_{seed}"


class ArrangePipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _seeds_for_dataset(self, dataset_name: str) -> Tuple[int, ...]:
        overrides = self.config.dataset_seed_overrides or {
            "tox21": (5, 6, 7, 8, 9),
            "sider": (5, 6, 7, 8, 9),
        }
        return overrides.get(dataset_name, self.config.seeds)

    def _reaction_config(self, spec, seed: int) -> ReactionFinetuneConfig:
        prefix = dataset_prefix(spec.name, seed)
        split_method = spec.split_method if self.config.split_method == "auto" else self.config.split_method
        return ReactionFinetuneConfig(
            pretrained_backbone_path=self.config.pretrained_backbone_path,
            output_dir=f"results/{prefix}/reaction",
            rxnfp_json_path=spec.rxnfp_json_path,
            sample_size=self.config.sample_size,
            seed=seed,
            model_init_seed=self.config.model_init_seed,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            epochs=self.config.reaction_epochs,
            learning_rate=self.config.learning_rate,
            backbone_learning_rate=self.config.backbone_learning_rate,
            split_method=split_method,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            device=self.config.device,
            use_absolute_branch=self.config.use_absolute_branch,
        )

    def _fusion_config(self, dataset_name: str, seed: int) -> FusionConfig:
        prefix = dataset_prefix(dataset_name, seed)
        return FusionConfig(
            reaction_checkpoint_path=f"results/{prefix}/reaction/best.pt",
            output_dir=f"results/{prefix}/fusion",
            model_init_seed=self.config.model_init_seed,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            epochs=self.config.fusion_epochs,
            learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            device=self.config.device,
            use_absolute_branch=self.config.use_absolute_branch,
        )

    def _append_result(self, row: dict) -> None:
        path = resolve_path(self.config.results_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        fieldnames = [
            "dataset",
            "seed",
            "stage",
            "metric",
            "best_val_score",
            "test_score",
            "checkpoint",
        ]
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or fieldnames)
                rows = list(reader)
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
        rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_one(self, dataset_name: str, seed: int) -> dict:
        spec = get_dataset_spec(dataset_name)
        reaction_cfg = self._reaction_config(spec, seed)
        dataset, split_indices = build_dataset(spec, reaction_cfg)
        print(
            f"\n===== {spec.name} seed={seed} split={reaction_cfg.split_method} "
            f"samples={len(dataset)} ====="
        )
        reaction_result = None
        if self.config.run_reaction_finetune:
            reaction_result = train_reaction_effect(
                spec,
                reaction_cfg,
                dataset=dataset,
                split_indices=split_indices,
            )
            self._append_result(
                {
                    "dataset": spec.name,
                    "seed": seed,
                    "stage": "reaction_finetune",
                    "metric": "delta_rmse",
                    "best_val_score": reaction_result["best_val_rmse"],
                    "test_score": reaction_result["best_test_rmse"],
                    "checkpoint": reaction_result["checkpoint"],
                }
            )

        fusion_result = None
        if self.config.run_fusion:
            fusion_cfg = self._fusion_config(spec.name, seed)
            if not Path(resolve_path(fusion_cfg.reaction_checkpoint_path)).exists():
                if reaction_result is None:
                    raise FileNotFoundError(
                        f"Fusion needs reaction checkpoint: {fusion_cfg.reaction_checkpoint_path}"
                    )
            fusion_result = train_fusion(
                spec,
                fusion_cfg,
                split_indices,
                dataset,
            )
            self._append_result(
                {
                    "dataset": spec.name,
                    "seed": seed,
                    "stage": "fusion",
                    "metric": fusion_result["metric"],
                    "best_val_score": fusion_result["best_val_score"],
                    "test_score": fusion_result["test_score"],
                    "checkpoint": str(resolve_path(fusion_cfg.output_dir) / "best.pt"),
                    **{
                        key: value
                        for key, value in fusion_result.items()
                        if key in {"test_cliff_rmse", "num_test", "num_test_cliff"}
                    },
                    **{key: value for key, value in fusion_result.items() if key.startswith("roc_auc_")},
                }
            )
        return {"reaction": reaction_result, "fusion": fusion_result}

    def run(self) -> None:
        for dataset_name in self.config.datasets:
            for seed in self._seeds_for_dataset(dataset_name):
                self.run_one(dataset_name, seed)
