import json
from tqdm import tqdm
import random
import os


def calculate_smiles_overlap_and_resplit(pretrain_path, sft_dirs, seed=42):
    random.seed(seed)

    # Read the set of SMILES from pretrain data
    total_lines = sum(1 for _ in open(pretrain_path, "r"))
    with open(pretrain_path, "r") as f:
        pretrain_smiles = set()
        for line in tqdm(f, total=total_lines, desc="Loading pretrain data"):
            data = json.loads(line)
            smiles = data.get("smiles")
            if smiles:
                pretrain_smiles.add(smiles)
    print(f"Total unique SMILES in pretrain set: {len(pretrain_smiles)}")

    # Process each SFT directory
    overlap_results = {}
    for sft_dir in tqdm(sft_dirs, desc="Processing SFT datasets"):
        merged_data = {}
        for split in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            split_path = os.path.join(sft_dir, split)
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        smiles = data.get("smiles")
                        if smiles:
                            merged_data[smiles] = (
                                data  # Deduplicate but keep complete fields
                            )

        merged_items = list(merged_data.items())
        sft_total = len(merged_items)
        sft_overlap = sum(1 for smi, _ in merged_items if smi in pretrain_smiles)
        overlap_ratio = sft_overlap / sft_total if sft_total > 0 else 0.0

        # Resplit into 8:1:1 ratio
        random.shuffle(merged_items)
        n_train = int(0.8 * sft_total)
        n_valid = int(0.1 * sft_total)
        train_set = merged_items[:n_train]
        valid_set = merged_items[n_train : n_train + n_valid]
        test_set = merged_items[n_train + n_valid :]

        def write_split(data_list, filename):
            path = os.path.join(sft_dir, filename)
            with open(path, "w") as f:
                for _, record in data_list:
                    json.dump(record, f)
                    f.write("\n")

        write_split(train_set, "train_dedup.jsonl")
        write_split(valid_set, "valid_dedup.jsonl")
        write_split(test_set, "test_dedup.jsonl")

        overlap_results[sft_dir] = {
            "sft_total": sft_total,
            "sft_overlap": sft_overlap,
            "overlap_ratio": round(overlap_ratio, 4),
        }

    return overlap_results


if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR")
    pretrain_file = os.path.join(data_dir, "dataset/pretrain/chembl_train_2d.jsonl")
    sft_files_moleculenet = [
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/bace_1/raw/"),
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/bbbp_1/raw/"),
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/clintox_1/raw/"),
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/delaney_1/raw/"),
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/freesolv_1/raw/"),
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/lipo_1/raw/"),
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/sider_1/raw/"),
        os.path.join(data_dir, "dataset/sft_geo_randomsplit/tox21_1/raw/"),
    ]

    sft_files_qm = [
        os.path.join(data_dir, "dataset/QM/qm7/"),
        os.path.join(data_dir, "dataset/QM/qm8/"),
        os.path.join(data_dir, "dataset/QM/qm9/"),
    ]

    sft_files_tdc = [
        os.path.join(data_dir, "dataset/sft_tdc_geo/BBB_Martins/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/Caco2_Wang/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/CYP2C9_Substrate_CarbonMangels/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/CYP2D6_Substrate_CarbonMangels/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/CYP3A4_Substrate_CarbonMangels/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/HIA_Hou/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/HydrationFreeEnergy_FreeSolv/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/Lipophilicity_AstraZeneca/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/PAMPA_NCATS/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/Pgp_Broccatelli/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/PPBR_AZ/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/Solubility_AqSolDB/"),
        os.path.join(data_dir, "dataset/tdc_geo_tox/AMES/"),
        os.path.join(data_dir, "dataset/tdc_geo_tox/DILI/"),
        os.path.join(data_dir, "dataset/tdc_geo_tox/hERG/"),
        os.path.join(data_dir, "dataset/tdc_geo_tox/LD50_Zhu/"),
        os.path.join(data_dir, "dataset/sft_tdc_geo/Bioavailability_Ma"),
    ]

    results = calculate_smiles_overlap_and_resplit(pretrain_file, sft_files_qm)
    # results = calculate_smiles_overlap_and_resplit(pretrain_file, sft_files_moleculenet)
    # results = calculate_smiles_overlap_and_resplit(pretrain_file, sft_files_tdc)

    for path, stats in results.items():
        print(f"{path}: ({stats['overlap_ratio']*100:.2f}%)")
