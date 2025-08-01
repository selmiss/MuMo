from preprocess.compound_tools import mol_to_geognn_graph_data_MMFF3d
from rdkit.Chem import AllChem
from rdkit import Chem
import os
import json
import numpy as np
from ogb.utils.mol import smiles2graph
from rdkit.Chem import Draw
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data, Batch
from preprocess.brics import bond_break
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import torch
from concurrent.futures import TimeoutError
import sys


def get_pretrain_bond_angle(edges, atom_poses):
    """tbd"""

    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle

    def _add_item(
        node_i_indices,
        node_j_indices,
        node_k_indices,
        bond_angles,
        node_i_index,
        node_j_index,
        node_k_index,
    ):
        node_i_indices += [node_i_index, node_k_index]
        node_j_indices += [node_j_index, node_j_index]
        node_k_indices += [node_k_index, node_i_index]
        pos_i = atom_poses[node_i_index]
        pos_j = atom_poses[node_j_index]
        pos_k = atom_poses[node_k_index]
        angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
        bond_angles += [angle, angle]

    E = len(edges)
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    bond_angles = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edges[edge_i]
            b0, b1 = edges[edge_j]
            if a0 == b0 and a1 == b1:
                continue
            if a0 == b1 and a1 == b0:
                continue
            if a0 == b0:
                _add_item(
                    node_i_indices,
                    node_j_indices,
                    node_k_indices,
                    bond_angles,
                    a1,
                    a0,
                    b1,
                )
            if a0 == b1:
                _add_item(
                    node_i_indices,
                    node_j_indices,
                    node_k_indices,
                    bond_angles,
                    a1,
                    a0,
                    b0,
                )
            if a1 == b0:
                _add_item(
                    node_i_indices,
                    node_j_indices,
                    node_k_indices,
                    bond_angles,
                    a0,
                    a1,
                    b1,
                )
            if a1 == b1:
                _add_item(
                    node_i_indices,
                    node_j_indices,
                    node_k_indices,
                    bond_angles,
                    a0,
                    a1,
                    b0,
                )
    node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)
    node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
    bond_angles = np.array(bond_angles)[uniq_index]
    return [node_i_indices, node_j_indices, node_k_indices, bond_angles]


def edge_transfer(edges):
    tn_edges = []
    for i in range(len(edges[0])):
        tn_edges.append([edges[0][i], edges[1][i]])
    return tn_edges


def build_bond_angle_graph(edge_index, node_i, node_j, node_k, bond_angles):
    graph = dict()
    num_nodes = len(edge_index[0])
    ba_node = np.zeros([num_nodes, 1])
    ba_edge_index = []
    index_start_list = []
    index_end_list = []
    ba_edge_feat = []
    for i in range(len(node_i)):
        bond1 = [node_i[i], node_j[i]]
        bond2 = [node_j[i], node_k[i]]
        angle = bond_angles[i]
        start_index = end_index = -1
        for j in range(len(edge_index[0])):
            if edge_index[0][j] == bond1[0] and edge_index[1][j] == bond1[1]:
                start_index = j
            elif edge_index[0][j] == bond2[0] and edge_index[1][j] == bond2[1]:
                end_index = j
        if start_index != -1 and end_index != -1:
            index_start_list.append(start_index)
            index_end_list.append(end_index)
            ba_edge_feat.append([angle])
    ba_edge_index.append(index_start_list)
    ba_edge_index.append(index_end_list)
    graph["num_nodes"] = num_nodes
    graph["node_feat"] = np.array(ba_node)
    graph["edge_feat"] = np.array(ba_edge_feat)
    graph["edge_index"] = np.array(ba_edge_index)
    return graph


def mol2GeoGraph(
    mol, brics: bool = True, geo_operation: bool = True, return_dict=False
):
    """
    Convert mol object with existing 3D coordinates to GeoGraph.

    Args:
        mol: RDKit mol object with conformer (already has 3D coordinates)
        brics: Whether to include BRICS fragmentation
        geo_operation: Whether to include geometric operations (bond angles)
        return_dict: Whether to return dictionary format

    Returns:
        Graph data structure similar to smiles2GeoGraph
    """

    if mol is None:
        return None

    # Get SMILES from mol for compatibility
    smiles = Chem.MolToSmiles(mol)

    # Build Atom-Bond Graph using existing smiles2graph function
    graph = smiles2graph(smiles)
    aba_graph = Data()
    aba_graph.__num_nodes__ = int(graph["num_nodes"])
    aba_graph.edge_index = graph["edge_index"]
    aba_graph.edge_attr = graph["edge_feat"]
    aba_graph.x = graph["node_feat"]

    # Build Bond-Angle Graph only if geo_operation is True
    if geo_operation:
        # Extract atom positions from existing conformer
        try:
            conf = mol.GetConformer()
            atom_pos = []
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                atom_pos.append([pos.x, pos.y, pos.z])
            atom_pos = np.array(atom_pos, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Could not extract atom positions from mol object: {e}")
            # Fallback to MMFF3d generation
            geo_data = mol_to_geognn_graph_data_MMFF3d(mol)
            atom_pos = geo_data["atom_pos"]

        tn_edges_index = edge_transfer(graph["edge_index"])
        node_i, node_j, node_k, bond_angles = get_pretrain_bond_angle(
            tn_edges_index, atom_pos
        )
        ba_graph = build_bond_angle_graph(
            graph["edge_index"],
            node_i=node_i,
            node_j=node_j,
            node_k=node_k,
            bond_angles=bond_angles,
        )

        # Merge 2 Graph into graph Data
        aba_graph.__num_ba_nodes__ = int(ba_graph["num_nodes"])
        aba_graph.ba_edge_index = ba_graph["edge_index"]
        aba_graph.ba_edge_attr = ba_graph["edge_feat"]

    # Get BRICS A-B Graph
    if brics:
        fra_edge_index, fra_edge_attr, cluster_idx = bond_break(mol)
        aba_graph.fra_edge_index = fra_edge_index
        aba_graph.fra_edge_attr = fra_edge_attr
        aba_graph.cluster_idx = cluster_idx

        # Get BRICS B-A GeoGraph only if geo_operation is True
        if geo_operation:
            tn_edges_index = edge_transfer(fra_edge_index)
            node_i, node_j, node_k, bond_angles = get_pretrain_bond_angle(
                tn_edges_index, atom_pos
            )
            ba_graph = build_bond_angle_graph(
                fra_edge_index,
                node_i=node_i,
                node_j=node_j,
                node_k=node_k,
                bond_angles=bond_angles,
            )
            aba_graph.bafra_edge_index = ba_graph["edge_index"]
            aba_graph.bafra_edge_attr = ba_graph["edge_feat"]

    aba_graph.smiles = smiles
    if return_dict:
        batch_data = Data(
            **{
                key: torch.tensor(value) if isinstance(value, list) else value
                for key, value in aba_graph.items()
            }
        )
        batch_data = Batch.from_data_list([batch_data])
        return aba_graph.to_dict()

    return aba_graph


def smiles2GeoGraph(
    smiles: str, brics: bool = True, geo_operation: bool = True, return_dict=False
):

    # Get mol from smiles.
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Build Atom-Bond Graph.
    graph = smiles2graph(smiles)
    aba_graph = Data()
    aba_graph.__num_nodes__ = int(graph["num_nodes"])
    aba_graph.edge_index = graph["edge_index"]
    aba_graph.edge_attr = graph["edge_feat"]
    aba_graph.x = graph["node_feat"]

    # Build Bond-Angle Graph only if geo_operation is True
    if geo_operation:
        geo_data = mol_to_geognn_graph_data_MMFF3d(mol)
        atom_pos = geo_data["atom_pos"]
        tn_edges_index = edge_transfer(graph["edge_index"])
        node_i, node_j, node_k, bond_angles = get_pretrain_bond_angle(
            tn_edges_index, atom_pos
        )
        ba_graph = build_bond_angle_graph(
            graph["edge_index"],
            node_i=node_i,
            node_j=node_j,
            node_k=node_k,
            bond_angles=bond_angles,
        )

        # Merge 2 Graph into graph Data
        aba_graph.__num_ba_nodes__ = int(ba_graph["num_nodes"])
        aba_graph.ba_edge_index = ba_graph["edge_index"]
        aba_graph.ba_edge_attr = ba_graph["edge_feat"]

    # Get BRICS A-B Graph
    if brics:
        fra_edge_index, fra_edge_attr, cluster_idx = bond_break(mol)
        aba_graph.fra_edge_index = fra_edge_index
        aba_graph.fra_edge_attr = fra_edge_attr
        aba_graph.cluster_idx = cluster_idx

        # Get BRICS B-A GeoGraph only if geo_operation is True
        if geo_operation:
            tn_edges_index = edge_transfer(fra_edge_index)
            node_i, node_j, node_k, bond_angles = get_pretrain_bond_angle(
                tn_edges_index, atom_pos
            )
            ba_graph = build_bond_angle_graph(
                fra_edge_index,
                node_i=node_i,
                node_j=node_j,
                node_k=node_k,
                bond_angles=bond_angles,
            )
            aba_graph.bafra_edge_index = ba_graph["edge_index"]
            aba_graph.bafra_edge_attr = ba_graph["edge_feat"]

    aba_graph.smiles = smiles
    if return_dict:
        batch_data = Data(
            **{
                key: torch.tensor(value) if isinstance(value, list) else value
                for key, value in aba_graph.items()
            }
        )
        batch_data = Batch.from_data_list([batch_data])
        return aba_graph.to_dict()

    return aba_graph


def preprocess_geo_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    if "smiles" not in df.columns:
        raise ValueError("The CSV file must contain a 'smiles' column.")
    metadata = df.drop(columns=["smiles"])
    graphs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
        smiles = row["smiles"]
        try:
            graph = smiles2GeoGraph(smiles, return_dict=True)
            graph.update(metadata.iloc[idx].to_dict())
            graphs.append(graph)
        except Exception as e:
            print(f"Error processing SMILES at index {idx}: {smiles}, error: {e}")

    graphs = [g for g in graphs if g is not None]
    with open(output_path, "w") as f:
        for row in tqdm(graphs):
            for key, value in row.items():
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    row[key] = value.tolist()

            json.dump(row, f)
            f.write("\n")
    print(f"Saved {len(graphs)} graph-data objects to {output_path}")


def process_smiles_to_graph_wrapper(args):
    idx, smiles, brics, geo_operation = args
    try:
        return idx, smiles2GeoGraph(
            smiles=smiles, brics=brics, geo_operation=geo_operation, return_dict=True
        )
    except Exception as e:
        print(str(e))
        return idx, None


def process_mol_to_graph_wrapper(args):
    """
    Wrapper function for multiprocessing mol objects with existing coordinates
    """
    idx, mol, brics, geo_operation = args
    try:
        return idx, mol2GeoGraph(
            mol=mol, brics=brics, geo_operation=geo_operation, return_dict=True
        )
    except Exception as e:
        print(f"Error processing mol at index {idx}: {str(e)}")
        return idx, None


def process_dataset_and_save_multithreaded(
    csv_file, output_file, num_workers=4, brics=True, geo_operation=True
):
    print(f"Starting processing with {num_workers} workers")
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    smiles_column = "smiles"
    if "smiles" not in df.columns:
        smiles_column = "mol"
        if "mol" not in df.columns:
            raise ValueError("The CSV file must contain a 'smiles' or 'mol' column.")

    print(f"Found {len(df)} SMILES to process")
    smiles_list = list(enumerate(df[smiles_column]))
    # Add brics and geo_operation parameters to each item in the list
    smiles_list = [(idx, smiles, brics, geo_operation) for idx, smiles in smiles_list]
    metadata = df.drop(columns=[smiles_column])
    results = [None] * len(df)

    print("Starting parallel processing...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_smiles_to_graph_wrapper, item)
            for item in smiles_list
        ]
        pbar = tqdm(
            total=len(smiles_list),
            desc="Processing SMILES",
            bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
            file=sys.stdout,
        )
        for future in futures:
            idx, graph = future.result()
            if graph is not None:
                graph.update(metadata.iloc[idx].to_dict())
            results[idx] = graph
            pbar.update(1)
        pbar.close()

    graphs = [g for g in results if g is not None]
    print(f"Successfully processed {len(graphs)} out of {len(df)} SMILES")

    print(f"Writing results to {output_file}")
    with open(output_file, "w") as f:
        pbar = tqdm(
            total=len(graphs),
            desc="Writing to JSONL",
            bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
            file=sys.stdout,
        )
        for row in graphs:
            for key, value in row.items():
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    row[key] = value.tolist()

            json.dump(row, f)
            f.write("\n")
            pbar.update(1)
        pbar.close()

    print(f"Saved {len(graphs)} graph-data objects to {output_file}")


# Usage example for mol2GeoGraph function:
"""
Example usage:

# For single mol object:
from rdkit import Chem
from rdkit.Chem import AllChem

# Load mol with existing 3D coordinates (e.g., from SDF file)
mol = Chem.MolFromSmilesWithCoords("CCO")  # or load from file
# Ensure the mol has 3D coordinates
if mol.GetNumConformers() == 0:
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

# Convert to graph
graph = mol2GeoGraph(mol, brics=True, geo_operation=True, return_dict=False)

# For batch processing:
mol_list = [mol1, mol2, mol3, ...]  # List of mol objects with coordinates
process_mol_dataset_and_save_multithreaded(
    mol_list=mol_list,
    output_file="output_graphs.jsonl",
    metadata_df=None,  # Optional metadata DataFrame
    num_workers=4,
    brics=True,
    geo_operation=True
)
"""


def process_mol_dataset_and_save_multithreaded(
    mol_list,
    output_file,
    metadata_df=None,
    num_workers=4,
    brics=True,
    geo_operation=True,
):
    """
    Process a list of mol objects with existing coordinates and save to JSONL format

    Args:
        mol_list: List of RDKit mol objects with conformers (already have 3D coordinates)
        output_file: Output JSONL file path
        metadata_df: Optional pandas DataFrame with metadata for each mol
        num_workers: Number of worker threads
        brics: Whether to include BRICS fragmentation
        geo_operation: Whether to include geometric operations
    """
    print(f"Starting processing with {num_workers} workers")
    print(f"Found {len(mol_list)} mol objects to process")

    # Prepare arguments for multiprocessing
    mol_args = [(idx, mol, brics, geo_operation) for idx, mol in enumerate(mol_list)]
    results = [None] * len(mol_list)

    print("Starting parallel processing...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_mol_to_graph_wrapper, item) for item in mol_args
        ]
        pbar = tqdm(
            total=len(mol_list),
            desc="Processing mol objects",
            bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
            file=sys.stdout,
        )
        for future in futures:
            idx, graph = future.result()
            if graph is not None and metadata_df is not None:
                graph.update(metadata_df.iloc[idx].to_dict())
            results[idx] = graph
            pbar.update(1)
        pbar.close()

    graphs = [g for g in results if g is not None]
    print(f"Successfully processed {len(graphs)} out of {len(mol_list)} mol objects")

    print(f"Writing results to {output_file}")
    with open(output_file, "w") as f:
        pbar = tqdm(
            total=len(graphs),
            desc="Writing to JSONL",
            bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
            file=sys.stdout,
        )
        for row in graphs:
            for key, value in row.items():
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    row[key] = value.tolist()

            json.dump(row, f)
            f.write("\n")
            pbar.update(1)
        pbar.close()

    print(f"Saved {len(graphs)} graph-data objects to {output_file}")


def check_3d_data_source(input_path):
    data = []
    with open(input_path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Parse each line as a JSON object
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

    for item in tqdm(data):
        item = {
            key: convert_value(key, value) if isinstance(value, list) else value
            for key, value in item.items()
        }

        item = Data(**item)
        print(item.cluster_idx)
        if not check_geo_data(item, type_map=type_map):
            print(item)
            import pdb

            pdb.set_trace()


def check_geo_data(data: Data, type_map: dict):
    # Check if edge indices and batch arrays are integers
    mark = True

    # Check keys.
    for item in type_map.keys():
        if item not in data.keys():
            print(f"Data Miss key {item}")
            mark = False

    # Check types.
    for key, value in data.items():
        if key in type_map and key != "smiles" and value.dtype != type_map[key]:
            print(
                f"Type error: key '{key}' type is {value.dtype}, expect {type_map[key]}"
            )
            mark = False

    # Check basic shape.
    try:
        if not (
            data.x.shape[1] == 9
            and data.x.shape[0] > 0
            and data.edge_attr.shape[1] == 3
            and data.ba_edge_attr.shape[1] == 1
            and data.fra_edge_attr.shape[1] == 3
            and data.cluster_idx.shape[0] == data.x.shape[0]
            and data.bafra_edge_attr.shape[1] == 1
        ):
            print(f"Error: The data shape is not right.")
            mark = False
            return mark
    except Exception as err:
        print(f"Error: The data shape is not right: {err}")
        mark = False
        return mark

    # Check edge dimention.
    if not (
        data.edge_index.shape[0] == 2
        and data.ba_edge_index.shape[0] == 2
        and data.fra_edge_index.shape[0] == 2
        and data.bafra_edge_index.shape[0] == 2
    ):
        print("Error: Edge dimention 0 are not 2.")
        mark = False

    # Check index of edges.
    if torch.max(data.edge_index) > data.x.shape[0] - 1:
        print(f"'edge_index' out of range, with max index {torch.max(data.edge_index)}")
        mark = False
    if torch.max(data.ba_edge_index) > data.edge_attr.shape[0] - 1:
        print(
            f"'ba_edge_index' out of range, with max index {torch.max(data.ba_edge_index)}"
        )
        mark = False
    if torch.max(data.fra_edge_index) > data.x.shape[0] - 1:
        print(
            f"'fra_edge_index' out of range, with max index {torch.max(data.fra_edge_index)}"
        )
        mark = False
    if torch.max(data.bafra_edge_index) > data.fra_edge_attr.shape[0] - 1:
        print(
            f"'bafra_edge_index' out of range, with max index {torch.max(data.bafra_edge_index)}"
        )
        mark = False
    return mark


def check_batch(data_batch):
    """
    Validate the integrity of a DataBatch object.

    Args:
        data_batch: A PyTorch Geometric DataBatch object containing graph data.

    Returns:
        bool: True if all checks pass, otherwise False.
    """
    # Check if edge indices and batch arrays are integers
    if not (
        data_batch.edge_index.dtype == torch.int64
        and data_batch.ba_edge_index.dtype == torch.int64
        and data_batch.fra_edge_index.dtype == torch.int64
        and data_batch.bafra_edge_index.dtype == torch.int64
        and data_batch.batch.dtype == torch.int64
    ):
        print("Error: Edge indices or batch array are not integers.")
        return False

    # Check if bond-angle nodes match atom-bond edges
    num_ba_nodes = data_batch.ba_edge_index.max().item() + 1
    num_ab_edges = data_batch.edge_index.size(1)
    # if num_ba_nodes != num_ab_edges:
    #     print(f"Error: Bond-angle nodes ({num_ba_nodes}) do not match atom-bond edges ({num_ab_edges}).")
    #     return False

    # Check if processed atom-bond edges are valid
    num_fra_edges = data_batch.fra_edge_index.size(1)
    if num_fra_edges > num_ab_edges:
        print(
            f"Error: Processed atom-bond edges ({num_fra_edges}) exceed original atom-bond edges ({num_ab_edges})."
        )
        return False

    # Check if the number of smiles matches the number of graphs
    num_graphs = data_batch.ptr.size(0) - 1
    if len(data_batch.smiles) != num_graphs:
        print(
            f"Error: Number of smiles ({len(data_batch.smiles)}) does not match number of graphs ({num_graphs})."
        )
        return False

    # All checks passed
    # print("All checks passed.")
    return True


def transfer_data(input_path, output_path):
    # Load your PyTorch dataset
    dataset = torch.load(input_path)

    # Save as JSONL
    with open(output_path, "w") as f:
        for row in tqdm(dataset):
            for key, value in row.items():
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    row[key] = value.tolist()

            json.dump(row, f)
            f.write("\n")


def generate_geotest_dataset_dir(input_root, output_root):

    for dirpath, dirnames, filenames in os.walk(input_root):
        output_dir = os.path.join(output_root, os.path.relpath(dirpath, input_root))
        os.makedirs(output_dir, exist_ok=True)

        for filename in filenames:
            if filename.endswith(".csv"):
                try:
                    print(f"Processing file {filename}")
                    input_file = os.path.join(dirpath, filename)
                    output_file = os.path.join(output_dir, filename)
                    process_dataset_and_save_multithreaded(
                        input_file, output_file, num_workers=10
                    )
                except Exception as err:
                    print(f"Error processing file {filename}: {str(err)}")


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


def merge_geotest_dataset_dir(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    all_json_objects = []
    for dirpath, dirnames, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.endswith(".csv"):
                jsonl_path = os.path.join(dirpath, filename)
                try:
                    # Read the JSONL file line by line
                    with open(jsonl_path, "r", encoding="utf-8") as file:
                        for line in file:
                            obj = json.loads(line)
                            # Filter the object to include only keys in type_map
                            filtered_obj = {
                                key: obj[key] for key in type_map.keys() if key in obj
                            }
                            all_json_objects.append(filtered_obj)

                except Exception as e:
                    print(f"Error processing file {jsonl_path}: {e}")
    output_jsonl_path = os.path.join(output_root, "merged_dataset.jsonl")
    try:
        # Write all JSON objects to a JSONL file
        with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for obj in all_json_objects:
                jsonl_file.write(json.dumps(obj) + "\n")

        print(f"Merged JSONL file created at: {output_jsonl_path}")
    except Exception as e:
        print(f"Error writing merged JSONL file: {e}")


if __name__ == "__main__":
    import os

    # Check for required environment variables
    if "DATA_DIR" not in os.environ:
        raise ValueError("Environment variable DATA_DIR not set")

    DATA_DIR = os.environ["DATA_DIR"]

    data_path = f"{DATA_DIR}/dataset/dk/targets/smiles_all_051.csv"
    output_path = f"{DATA_DIR}/dataset/dk/targets/smiles_all_051.jsonl"
    process_dataset_and_save_multithreaded(
        data_path, output_path, num_workers=10, brics=False, geo_operation=False
    )

    # transfer_data(input_path=output_path, output_path=f"{DATA_DIR}/dataset/pretrain/geo_data/chembl_train_dict.jsonl")

    # test_data_dir = f"{DATA_DIR}/dataset/test_randomsplit/bace"
    # test_data_output_dir = f"{DATA_DIR}/dataset/test_geo_randomsplit/bace"
    # generate_geotest_dataset_dir(test_data_dir, test_data_output_dir)

    # test_data_dir = f"{DATA_DIR}/dataset/test_scaffoldsplit"
    # test_data_output_dir = f"{DATA_DIR}/dataset/test_geo_scaffoldsplit"
    # generate_geotest_dataset_dir(test_data_dir, test_data_output_dir)

    # input_path = f"{DATA_DIR}/dataset/test_geo/sider_1/raw/train_sider_1.csv"
    # check_3d_data_source(input_path)

    # output_file = f"{DATA_DIR}/dataset/pretrain/geo_data/test_merge.jsonl"
    # merge_geotest_dataset_dir(test_data_output_dir, output_root=output_file)
