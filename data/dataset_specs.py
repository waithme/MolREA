from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


TOX21_TASKS = (
    "NR-AhR",
    "NR-AR-LBD",
    "NR-AR",
    "NR-Aromatase",
    "NR-ER-LBD",
    "NR-ER",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


SIDER_TASKS = (
    "Blood and lymphatic system disorders",
    "Cardiac disorders",
    "Congenital, familial and genetic disorders",
    "Ear and labyrinth disorders",
    "Endocrine disorders",
    "Eye disorders",
    "Gastrointestinal disorders",
    "General disorders and administration site conditions",
    "Hepatobiliary disorders",
    "Immune system disorders",
    "Infections and infestations",
    "Injury, poisoning and procedural complications",
    "Investigations",
    "Metabolism and nutrition disorders",
    "Musculoskeletal and connective tissue disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Nervous system disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Product issues",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Reproductive system and breast disorders",
    "Respiratory, thoracic and mediastinal disorders",
    "Skin and subcutaneous tissue disorders",
    "Social circumstances",
    "Surgical and medical procedures",
    "Vascular disorders",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: str  # regression, binary, multitask_binary
    data_path: str
    reaction_json_path: str
    rxnfp_json_path: str
    id_column: str
    smiles_column: str
    target_columns: Tuple[str, ...]
    metric_names: Tuple[str, ...]
    result_metric: str
    proxy_kind: str
    split_method: str = "scaffold"


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "chembl228_ki": DatasetSpec(
        name="chembl228_ki",
        task_type="regression",
        data_path="data/dataset/CHEMBL228_Ki/CHEMBL228_Ki.csv",
        reaction_json_path="data/dataset/CHEMBL228_Ki/CHEMBL228_Ki_reaction.json",
        rxnfp_json_path="data/dataset/CHEMBL228_Ki/CHEMBL228_Ki_rxnfp.json",
        id_column="CID",
        smiles_column="smiles",
        target_columns=("y",),
        metric_names=("CHEMBL228_Ki",),
        result_metric="rmse",
        proxy_kind="chembl228_ki",
        split_method="official_cliff",
    ),
    "chembl234_ki": DatasetSpec(
        name="chembl234_ki",
        task_type="regression",
        data_path="data/dataset/CHEMBL234_Ki/CHEMBL234_Ki.csv",
        reaction_json_path="data/dataset/CHEMBL234_Ki/CHEMBL234_Ki_reaction.json",
        rxnfp_json_path="data/dataset/CHEMBL234_Ki/CHEMBL234_Ki_rxnfp.json",
        id_column="CID",
        smiles_column="smiles",
        target_columns=("y",),
        metric_names=("CHEMBL234_Ki",),
        result_metric="rmse",
        proxy_kind="chembl234_ki",
        split_method="official_cliff",
    ),
    "chembl239_ec50": DatasetSpec(
        name="chembl239_ec50",
        task_type="regression",
        data_path="data/dataset/CHEMBL239_EC50/CHEMBL239_EC50.csv",
        reaction_json_path="data/dataset/CHEMBL239_EC50/CHEMBL239_EC50_reaction.json",
        rxnfp_json_path="data/dataset/CHEMBL239_EC50/CHEMBL239_EC50_rxnfp.json",
        id_column="CID",
        smiles_column="smiles",
        target_columns=("y",),
        metric_names=("CHEMBL239_EC50",),
        result_metric="rmse",
        proxy_kind="chembl239_ec50",
        split_method="official_cliff",
    ),
    "chembl244_ki": DatasetSpec(
        name="chembl244_ki",
        task_type="regression",
        data_path="data/dataset/CHEMBL244_Ki/CHEMBL244_Ki.csv",
        reaction_json_path="data/dataset/CHEMBL244_Ki/CHEMBL244_Ki_reaction.json",
        rxnfp_json_path="data/dataset/CHEMBL244_Ki/CHEMBL244_Ki_rxnfp.json",
        id_column="CID",
        smiles_column="smiles",
        target_columns=("y",),
        metric_names=("CHEMBL244_Ki",),
        result_metric="rmse",
        proxy_kind="chembl244_ki",
        split_method="official_cliff",
    ),
    "chembl2147_ki": DatasetSpec(
        name="chembl2147_ki",
        task_type="regression",
        data_path="data/dataset/CHEMBL2147_Ki/CHEMBL2147_Ki.csv",
        reaction_json_path="data/dataset/CHEMBL2147_Ki/CHEMBL2147_Ki_reaction.json",
        rxnfp_json_path="data/dataset/CHEMBL2147_Ki/CHEMBL2147_Ki_rxnfp.json",
        id_column="CID",
        smiles_column="smiles",
        target_columns=("y",),
        metric_names=("PIM1_Ki",),
        result_metric="rmse",
        proxy_kind="chembl2147_ki",
        split_method="official_cliff",
    ),
    "esol": DatasetSpec(
        name="esol",
        task_type="regression",
        data_path="data/dataset/esol/esol.json",
        reaction_json_path="data/dataset/esol/esol_reaction.json",
        rxnfp_json_path="data/dataset/esol/esol_rxnfp.json",
        id_column="CID",
        smiles_column="SMILES",
        target_columns=("measured",),
        metric_names=("ESOL",),
        result_metric="rmse",
        proxy_kind="esol",
    ),
    "freesolve": DatasetSpec(
        name="freesolve",
        task_type="regression",
        data_path="data/dataset/freesolve/freesolve.json",
        reaction_json_path="data/dataset/freesolve/freesolve_reaction.json",
        rxnfp_json_path="data/dataset/freesolve/freesolve_rxnfp.json",
        id_column="CID",
        smiles_column="SMILES",
        target_columns=("measured",),
        metric_names=("FreeSolv",),
        result_metric="rmse",
        proxy_kind="freesolve",
    ),
    "lip": DatasetSpec(
        name="lip",
        task_type="regression",
        data_path="data/dataset/lip/lip.json",
        reaction_json_path="data/dataset/lip/lip_reaction.json",
        rxnfp_json_path="data/dataset/lip/lip_rxnfp.json",
        id_column="CID",
        smiles_column="SMILES",
        target_columns=("measured",),
        metric_names=("Lipophilicity",),
        result_metric="rmse",
        proxy_kind="lip",
    ),
    "bace": DatasetSpec(
        name="bace",
        task_type="binary",
        data_path="data/dataset/bace/bace.json",
        reaction_json_path="data/dataset/bace/bace_reaction.json",
        rxnfp_json_path="data/dataset/bace/bace_rxnfp.json",
        id_column="CID",
        smiles_column="SMILES",
        target_columns=("class",),
        metric_names=("BACE",),
        result_metric="roc_auc",
        proxy_kind="bace",
    ),
    "bbbp": DatasetSpec(
        name="bbbp",
        task_type="binary",
        data_path="data/dataset/bbbp/bbbp.json",
        reaction_json_path="data/dataset/bbbp/bbbp_reaction.json",
        rxnfp_json_path="data/dataset/bbbp/bbbp_rxnfp.json",
        id_column="CID",
        smiles_column="SMILES",
        target_columns=("class",),
        metric_names=("BBBP",),
        result_metric="roc_auc",
        proxy_kind="bbbp",
    ),
    "tox21": DatasetSpec(
        name="tox21",
        task_type="multitask_binary",
        data_path="data/dataset/tox21/tox21.json",
        reaction_json_path="data/dataset/tox21/tox21_reaction.json",
        rxnfp_json_path="data/dataset/tox21/tox21_rxnfp.json",
        id_column="mol_id",
        smiles_column="SMILES",
        target_columns=TOX21_TASKS,
        metric_names=TOX21_TASKS,
        result_metric="roc_auc",
        proxy_kind="tox21",
    ),
    "sider": DatasetSpec(
        name="sider",
        task_type="multitask_binary",
        data_path="data/dataset/sider/sider.json",
        reaction_json_path="data/dataset/sider/sider_reaction.json",
        rxnfp_json_path="data/dataset/sider/sider_rxnfp.json",
        id_column="CID",
        smiles_column="SMILES",
        target_columns=SIDER_TASKS,
        metric_names=SIDER_TASKS,
        result_metric="roc_auc",
        proxy_kind="sider",
    ),
}


def resolve_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [
        PACKAGE_DIR / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_SPECS)}")
    return DATASET_SPECS[key]
