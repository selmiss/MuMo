import logging
import os
import sys
import pandas as pd
import joblib
import torch
import numpy as np
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoTokenizer
from model.load_model import load_model
from torch_geometric.data import Data
from graph_batch import TriBatch
from preprocess.mol3d_processor import smiles2GeoGraph
from tqdm import tqdm

logger = logging.getLogger(__name__)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with MuMo model")
    # Model loading arguments
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--model_class", type=str, default="MuMoFinetune", help="Model class to use"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        help="Task type (regression/classification)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Where to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Will use the token generated when running huggingface-cli login",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        type=str,
        default="true",
        help="Whether to use fast tokenizer (true/false)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Override the default torch.dtype",
    )
    parser.add_argument(
        "--output_size", type=int, default=-1, help="Specify the model's output size"
    )
    parser.add_argument("--pool_method", type=str, help="Pooling method for the model")

    # Data arguments
    parser.add_argument(
        "--test_files", type=str, required=True, help="Path to test data file"
    )
    parser.add_argument(
        "--data_column_name",
        type=str,
        default="smiles",
        help="Name of the input column",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--scaler_path", type=str, help="Path to scaler file for inverse transformation"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save predictions"
    )

    return parser.parse_args()


def process_smiles_to_graph(smiles_list):
    """Process a single SMILES string to graph data"""
    output_smiles = []
    output_x = []
    output_edge_index = []
    output_edge_attr = []
    for smiles in smiles_list:
        graph_data = smiles2GeoGraph(smiles, brics=False, geo_operation=False)
        output_smiles.append(smiles)
        output_x.append(graph_data.x.tolist())
        output_edge_index.append(graph_data.edge_index.tolist())
        output_edge_attr.append(graph_data.edge_attr.tolist())
    return {
        "smiles": output_smiles,
        "x": output_x,
        "edge_index": output_edge_index,
        "edge_attr": output_edge_attr,
    }


def process_batch_for_inference(batch, tokenizer, max_length):
    from preprocess.mol3d_processor import smiles2GeoGraph

    """Process a batch of data for inference, similar to custom_collate_fn in finetuning"""

    def tokenize_and_batch(inputs):
        smiles = inputs["smiles"]  # Already a list of SMILES strings
        tokenized = tokenizer(
            smiles,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def batch_graph_data(inputs):
        def convert_value(key, value):
            try:
                if key in type_map:
                    converter = type_map[key]
                    if callable(converter):
                        return [converter(v) for v in value]
                    else:
                        return [torch.tensor(v, dtype=converter) for v in value]
                return value
            except Exception as e:
                print(
                    f"Error converting key '{key}' with value type {type(value)}: {str(e)}"
                )
                raise e

        def ensure_list_of_dict(batch_dict):
            keys = batch_dict.keys()
            batch_size = len(next(iter(batch_dict.values())))
            return [{k: batch_dict[k][i] for k in keys} for i in range(batch_size)]

        # Create Data object with error handling
        try:
            data_dict = {}
            for key, value in inputs.items():
                if isinstance(value, list):
                    data_dict[key] = convert_value(key, value)

            # Convert to list of dict
            list_of_dict = ensure_list_of_dict(data_dict)

            # Construct list of Data objects
            batch_data = [Data(**sample_dict) for sample_dict in list_of_dict]

            graph_batch = TriBatch.from_data_list(batch_data)
            return graph_batch
        except Exception as e:
            print(f"Error in batch_graph_data: {str(e)}")
            print(f"Input keys: {list(inputs.keys())}")
            raise e

    tokenized_batch = tokenize_and_batch(batch)
    input_ids = tokenized_batch[0]
    attention_mask = tokenized_batch[1]
    graph_data = batch_graph_data(batch)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "graph_data": graph_data,
    }


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "use_auth_token": True if args.use_auth_token else None,
    }

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer == "true",
        "revision": args.model_revision,
        "use_auth_token": True if args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, **tokenizer_kwargs
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model
    model = load_model(config, tokenizer, model_args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use DataParallel if multiple GPUs are available
    model.eval()  # Set to evaluation mode

    logger.info(f"Using device: {device}")

    # Load test data
    file_extension = os.path.splitext(args.test_files)[1].lower()
    format_type = "csv" if file_extension == ".csv" else "json"

    raw_datasets = load_dataset(
        format_type,
        data_files={"test": args.test_files},
    )
    test_dataset = raw_datasets["test"]

    # Process data in batches
    all_predictions = None
    all_smiles = []  # Rename to be more clear
    all_original_data = []  # Store original data to preserve other columns

    # Process data in batches
    total_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size

    for i in tqdm(
        range(0, len(test_dataset), args.batch_size),
        total=total_batches,
        desc="Processing batches",
    ):
        batch = test_dataset[i : i + args.batch_size]

        # Store original batch data to preserve other columns
        if isinstance(batch, dict):
            # Convert batch dict to list of dicts for easier handling
            batch_size = len(batch[list(batch.keys())[0]])
            original_batch_data = []
            for j in range(batch_size):
                item_data = {}
                for key, value in batch.items():
                    if isinstance(value, list) and j < len(value):
                        item_data[key] = value[j]
                    elif not isinstance(value, list):
                        item_data[key] = value
                original_batch_data.append(item_data)
        else:
            original_batch_data = batch

        # Process each item in the batch to generate graph data
        processed_batch = []

        # Check if batch already contains graph data
        if isinstance(batch, dict) and "x" in batch:
            processed_batch = batch
        elif isinstance(batch, dict) and "smiles" in batch:
            smiles = batch["smiles"]
            processed_batch = process_smiles_to_graph(smiles)
        else:
            continue

        if not processed_batch:
            continue

        batch_data = process_batch_for_inference(
            processed_batch, tokenizer, args.max_length
        )

        # Move tensors to device
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(device)
            elif hasattr(value, "to"):  # For graph data
                batch_data[key] = value.to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**batch_data)
            predictions = outputs["logits"].cpu().numpy()
            logger.debug(f"Batch predictions shape: {predictions.shape}")
            logger.debug(f"First few predictions: {predictions[:5]}")

            # Store predictions
            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions = np.concatenate([all_predictions, predictions])

            # Store SMILES for this batch
            all_smiles.extend(processed_batch["smiles"])

            # Store original data for this batch
            all_original_data.extend(original_batch_data)

    logger.debug(f"Total predictions shape: {all_predictions.shape}")
    logger.debug(f"Total SMILES: {len(all_smiles)}")

    # Load scaler if provided
    if args.scaler_path:
        scaler = joblib.load(args.scaler_path)
        predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
    else:
        predictions = (
            all_predictions.flatten()
        )  # Make sure predictions are 1-dimensional

    # Save final results
    # Create DataFrame with all original columns plus predictions
    logger.info(
        f"Creating final DataFrame with {len(all_original_data)} samples and {len(predictions)} predictions"
    )

    if len(all_original_data) != len(predictions):
        logger.warning(
            f"Mismatch between original data ({len(all_original_data)}) and predictions ({len(predictions)})"
        )
        # Use the shorter length to avoid index errors
        min_length = min(len(all_original_data), len(predictions))
        all_original_data = all_original_data[:min_length]
        predictions = predictions[:min_length]

    result_data = []
    for i, original_item in enumerate(all_original_data):
        item_data = original_item.copy()  # Copy all original columns
        item_data["prediction"] = predictions[i]  # Add prediction
        result_data.append(item_data)

    df = pd.DataFrame(result_data)
    logger.info(f"Final DataFrame columns: {list(df.columns)}")
    output_path = os.path.join(args.output_dir, "test_predictions.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
