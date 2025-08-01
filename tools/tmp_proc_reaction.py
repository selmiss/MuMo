from torch_geometric.data import Data
from train.graph_batch import TriBatch
import torch
import json
from tqdm import tqdm
import os

type_map = {
    "x": torch.float32,
    "edge_index": torch.int64,
    "edge_attr": torch.float32,
    "ba_edge_index": torch.int64,
    "ba_edge_attr": torch.float32,
    "fra_edge_index": torch.int64,
    "fra_edge_attr": torch.float32,
    "cluster_idx": torch.int64,
    "bafra_edge_index": torch.int64,
    "bafra_edge_attr": torch.float32,
    "smiles": str,
}


def convert_value(key, value):
    if key in type_map:
        converter = type_map[key]
        if callable(converter):
            return converter(value)
        else:
            return torch.tensor(value, dtype=converter)
    return value


def process_dataset(input_path, output_path, target_column_name="label"):
    raw_datasets = []
    with open(input_path, "r") as f:
        for line in f:
            raw_datasets.append(json.loads(line.strip()))

    result = []
    for item in tqdm(raw_datasets):
        labels = item[target_column_name]
        smiles = " ".join([mol["smiles"] for mol in item["mol_list"]])

        batch_data = [
            Data(
                **{
                    key: convert_value(key, value) if isinstance(value, list) else value
                    for key, value in mol_dict.items()
                }
            )
            for mol_dict in item["mol_list"]
        ]
        graph_uni = TriBatch.from_data_list(batch_data)

        data = {}

        for key in graph_uni.keys():
            if key == "batch":
                continue
            data[key] = graph_uni[key].tolist()

        data["smiles"] = smiles
        data[target_column_name] = labels
        result.append(data)

    with open(output_path, "w") as f:
        for data in result:
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    DATA_DIR = os.getenv("DATA_DIR")
    name = "USPTO"
    process_dataset(
        f"{DATA_DIR}/dataset/reaction_yield/{name}/{name}.jsonl",
        f"{DATA_DIR}/dataset/reaction_yield/{name}/{name}_processed.jsonl",
    )
