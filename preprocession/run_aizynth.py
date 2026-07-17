import json
from pathlib import Path
import csv
from aizynthfinder.aizynthfinder import AiZynthExpander


# --- 配置区域 ---
CONFIG_PATH = "../chem_data/config.yml"      # 确保这是绝对路径或相对于当前脚本的路径
INPUT_CSV = Path("./data/refined_HIV.csv")
OUTPUT_DIR = Path(".")                       # 结果 JSON 输出目录（当前目录）


def process_dataset(expander: AiZynthExpander, input_csv: Path) -> None:
    dataset_name = input_csv.stem.replace("refined_", "")
    output_path = OUTPUT_DIR / f"{dataset_name}_reaction.json"

    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row.get("CID", "")).strip()
            smiles = str(row.get("SMILES", "")).strip()
            if cid and smiles:
                rows.append((cid, smiles))

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}  共 {len(rows)} 个分子")
    print(f"{'='*60}")

    error = 0
    reactants_info = {}

    for i, (cid, smiles) in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] 正在处理: {cid} -> {smiles}")

        try:
            reactions = expander.do_expansion(smiles)
            molecule_reactions_list = []
            for reaction_tuple in reactions:
                temp = [[mol.smiles for mol in reaction_tuple[0].reactants[0]]]
                temp.append(reaction_tuple[0].metadata)
                molecule_reactions_list.append(temp)
            reactants_info[cid] = molecule_reactions_list
        except Exception as e:
            print(f"    错误: 处理分子 {smiles} 时崩溃。原因: {e}")
            error += 1
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reactants_info, f, indent=4, ensure_ascii=False)

    success_count = len(rows) - error
    print(f"已保存 → {output_path}")
    print(f"成功: {success_count}  失败: {error}  共: {len(rows)}")


def run_batch():
    # 1. 初始化 AiZynthExpander（只加载一次，所有数据集共用）
    try:
        print(f"正在加载配置: {CONFIG_PATH}...")
        expander = AiZynthExpander(configfile=CONFIG_PATH)
        expander.expansion_policy.select("uspto")
        expander.filter_policy.select("uspto")
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        return

    # 2. 检查输入 CSV
    if not INPUT_CSV.exists():
        print(f"未找到输入 CSV: {INPUT_CSV}")
        return

    print("将处理输入文件:")
    print(f"  {INPUT_CSV}")

    # 3. 处理 CSV
    process_dataset(expander, INPUT_CSV)

    print("\n处理完毕。")


if __name__ == "__main__":
    run_batch()
