from preprocess.mol3d_processor import smiles2GeoGraph
import pandas as pd
import json
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import os


def merge_two_graphs(graph_data1, graph_data2):
    # 1. Node count offset
    num_nodes1 = graph_data1.x.shape[0]

    # 2. Renumber edge_index of graph2
    edge_index2 = graph_data2.edge_index + num_nodes1

    # 3. Merge node features
    x = np.concatenate([graph_data1.x, graph_data2.x], axis=0)
    edge_index = np.concatenate([graph_data1.edge_index, edge_index2], axis=1)
    edge_attr = np.concatenate([graph_data1.edge_attr, graph_data2.edge_attr], axis=0)

    # 5. Other fields can also be merged if needed, such as batch, pos, etc.

    # 6. Construct new Data object
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def process_multi_smiles_to_2d_graph(example, num_input_smiles, label_key="Output"):

    mol_list = []
    result = {}
    label = example[label_key]

    for i in range(num_input_smiles):
        smiles = example["smiles_" + str(i)]

        components = smiles.split(",")

        if len(components) == 1:
            graph_data = smiles2GeoGraph(smiles, brics=False, geo_operation=False)
        else:
            graph_data = smiles2GeoGraph(
                components[0], brics=False, geo_operation=False
            )
            for component in components[1:]:
                graph_data_1 = smiles2GeoGraph(
                    component, brics=False, geo_operation=False
                )
                if graph_data_1 is None:
                    return None
                graph_data = merge_two_graphs(graph_data, graph_data_1)

        if graph_data is None:
            return None

        # Preserve all other columns (including labels)
        graph_dict = {
            "smiles": smiles,
            "x": graph_data.x.tolist(),
            "edge_index": graph_data.edge_index.tolist(),
            "edge_attr": graph_data.edge_attr.tolist(),
        }
        mol_list.append(graph_dict)

    result = {
        "mol_list": mol_list,
        "label": label,
    }

    return result


def process_multi_smiles_to_graph(example, num_input_smiles, label_key="Output"):

    mol_list = []
    result = {}
    label = example[label_key]

    for i in range(num_input_smiles):
        smiles = example["smiles_" + str(i)]
        graph_data = smiles2GeoGraph(smiles, brics=True, geo_operation=True)
        if graph_data is None:
            return None
        # Preserve all other columns (including labels)
        graph_dict = {
            "smiles": smiles,
            "x": graph_data.x.tolist(),
            "edge_index": graph_data.edge_index.tolist(),
            "edge_attr": graph_data.edge_attr.tolist(),
            "ba_edge_index": graph_data.ba_edge_index.tolist(),
            "ba_edge_attr": graph_data.ba_edge_attr.tolist(),
            "fra_edge_index": graph_data.fra_edge_index.tolist(),
            "fra_edge_attr": graph_data.fra_edge_attr.tolist(),
            "bafra_edge_index": graph_data.bafra_edge_index.tolist(),
            "bafra_edge_attr": graph_data.bafra_edge_attr.tolist(),
            "cluster_idx": (
                graph_data.cluster_idx.tolist()
                if type(graph_data.cluster_idx) == torch.Tensor
                else graph_data.cluster_idx
            ),
        }
        mol_list.append(graph_dict)

    result = {
        "mol_list": mol_list,
        "label": label,
    }

    return result


def main():
    DATA_DIR = os.getenv("DATA_DIR")
    name = "USPTO"
    input_path = f"{DATA_DIR}/dataset/reaction_yield/{name}/{name}.csv"
    output_path = f"{DATA_DIR}/dataset/reaction_yield/{name}/{name}.jsonl"

    df = pd.read_csv(input_path, header=0)
    result_list = []
    print(f"Processing {len(df)} rows")
    total = len(df)

    count = 0
    for index, row in tqdm(df.iterrows(), total=total):
        result = process_multi_smiles_to_2d_graph(row, 5, "label")
        result_list.append(result)
        count += 1
    print(f"Processed {count} rows, {count/total*100:.2f}%")
    with open(output_path, "w") as f:
        for result in result_list:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
