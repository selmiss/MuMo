from rdkit import Chem
import pandas as pd
import warnings
import json
import os
from tqdm import tqdm
from preprocess.mol3d_processor import mol2GeoGraph


def get_label_keys(graph_data):
    """
    Helper function to extract label keys from graph data (excluding graph structure keys)

    Args:
        graph_data: Dictionary containing graph data and labels

    Returns:
        list: List of keys that are labels (not graph structure)
    """
    # Graph structure keys to exclude
    structure_keys = {
        "smiles",
        "molecule_id",
        "x",
        "edge_index",
        "edge_attr",
        "__num_nodes__",
        "ba_edge_index",
        "ba_edge_attr",
        "__num_ba_nodes__",
        "fra_edge_index",
        "fra_edge_attr",
        "cluster_idx",
        "bafra_edge_index",
        "bafra_edge_attr",
    }

    return [k for k in graph_data.keys() if k not in structure_keys]


def construct_qm_data(file_path):
    """
    Process QM7 dataset by reading SDF file and corresponding CSV labels

    Args:
        file_path: Path to SDF file (CSV file should be at file_path + ".csv")

    Returns:
        list: List of graph data dictionaries with each CSV column as an individual key
    """

    supplier = Chem.SDMolSupplier(file_path)
    label_csv = pd.read_csv(file_path + ".csv", header=0)

    print(f"CSV file contains {label_csv.shape[1]} label columns")
    print(f"CSV column names: {list(label_csv.columns)}")
    print(f"len(supplier): {len(supplier)}")
    print(f"len(label_csv): {len(label_csv)}")
    # none_count = 0
    # for idx, mol in enumerate(tqdm(supplier, desc="Checking molecules", total=len(supplier))):
    #     if mol is None:
    #         warnings.warn(f"Mol is None. Index: {idx}.")
    #         none_count += 1
    #         continue
    # print(f"none_count: {none_count}")

    # exit()
    graphs = []  # List to store all graph data
    successful_count = 0

    for idx, mol in enumerate(
        tqdm(supplier, desc="Processing molecules", total=len(label_csv))
    ):
        if mol is None:
            # warnings.warn(f"Mol is None. Index: {idx}.")
            continue

        # Check if we have corresponding labels for this molecule
        if idx >= len(label_csv):
            warnings.warn(f"No labels found for molecule at index {idx}. Skipping.")
            continue

        # Convert mol to graph using mol2GeoGraph
        try:
            graph = mol2GeoGraph(mol, brics=True, geo_operation=True, return_dict=True)
        except Exception as e:
            warnings.warn(
                f"Error converting molecule at index {idx} to graph: {e}. Skipping."
            )
            continue

        if graph is None:
            warnings.warn(
                f"Failed to convert molecule at index {idx} to graph. Skipping."
            )
            continue

        # Add all labels from CSV row using actual column names
        labels = {}
        for col_name in label_csv.columns:
            labels[col_name] = label_csv.iloc[idx][col_name]

        # Add molecule id
        labels["molecule_id"] = mol.GetProp("_Name")

        # Update graph with all labels
        graph.update(labels)
        graphs.append(graph)
        successful_count += 1

    print(
        f"Successfully processed {successful_count} out of {len(label_csv)} molecules"
    )

    # Save results to JSONL file in the same directory
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_graphs.jsonl")

    print(f"Saving results to: {output_file}")

    with open(output_file, "w") as f:
        for graph_data in tqdm(graphs, desc="Writing to JSONL"):
            # Convert numpy arrays and tensors to lists for JSON serialization
            serializable_data = {}
            for key, value in graph_data.items():
                if hasattr(value, "tolist"):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value

            json.dump(serializable_data, f)
            f.write("\n")

    print(f"Saved {len(graphs)} molecular graphs to {output_file}")
    return graphs


def main():
    DATA_DIR = os.getenv("DATA_DIR")
    # Example usage of the original function
    print("=== Using construct_qm_data ===")
    graphs_list = construct_qm_data(f"{DATA_DIR}/dataset/QM/ori/gdb9/gdb9.sdf")

    # Process first few molecules as example
    print(f"Total graphs generated: {len(graphs_list)}")
    for i in range(min(3, len(graphs_list))):  # Show first 3 molecules as example
        graph_data = graphs_list[i]
        print(f"Molecule {i}: {list(graph_data.keys())}")
        # Show individual label values
        label_keys = get_label_keys(graph_data)
        print(f"Label keys: {label_keys}")
        for key in label_keys[:3]:  # Show first 3 labels
            print(f"  {key}: {graph_data.get(key, 'No value')}")
        print("---")


if __name__ == "__main__":
    main()
