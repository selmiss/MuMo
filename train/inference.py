import logging
import os
import sys
import pandas as pd
import joblib
import torch
import argparse
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from model.load_model import load_model
from torch_geometric.data import Data
from graph_batch import TriBatch

logger = logging.getLogger(__name__)

def convert_value(key, value, type_map):
    """Convert value to appropriate tensor type based on type_map."""
    if key in type_map:
        converter = type_map[key]
        if callable(converter):
            return converter(value)
        else:
            return torch.tensor(value, dtype=converter)
    return value

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with MuMo model")
    # Model loading arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--model_class", type=str, default="MuMoFinetune", help="Model class to use")
    parser.add_argument("--task_type", type=str, default="regression", help="Task type (regression/classification)")
    parser.add_argument("--config_name", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--cache_dir", type=str, help="Where to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--model_revision", type=str, default="main", help="The specific model version to use")
    parser.add_argument("--use_auth_token", action="store_true", help="Will use the token generated when running huggingface-cli login")
    parser.add_argument("--use_fast_tokenizer", type=str, default="true", help="Whether to use fast tokenizer (true/false)")
    parser.add_argument("--torch_dtype", type=str, choices=["auto", "bfloat16", "float16", "float32"], help="Override the default torch.dtype")
    parser.add_argument("--output_size", type=int, default=-1, help="Specify the model's output size")
    parser.add_argument("--pool_method", type=str, help="Pooling method for the model")
    
    # Data arguments
    parser.add_argument("--test_files", type=str, required=True, help="Path to test data file")
    parser.add_argument("--data_column_name", type=str, default="smiles", help="Name of the input column")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--scaler_path", type=str, help="Path to scaler file for inverse transformation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    
    return parser.parse_args()

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
        args.model_name_or_path,
        **tokenizer_kwargs
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load model
    model = load_model(config, tokenizer, model_args=args)
    model.eval()  # Set to evaluation mode
    
    # Load test data
    raw_datasets = load_dataset(
        "json",  # or "csv" depending on your file format
        data_files={"test": args.test_files},
    )
    test_dataset = raw_datasets["test"]
    
    # Type mapping for graph data
    type_map = {
        "x": torch.float32,
        "edge_index": torch.int64,
        "edge_attr": torch.float32,
        'ba_edge_index': torch.int64, 
        'ba_edge_attr': torch.float32, 
        'fra_edge_index': torch.int64, 
        'fra_edge_attr': torch.float32, 
        'cluster_idx': torch.int64, 
        'bafra_edge_index': torch.int64, 
        'bafra_edge_attr': torch.float32,
        "smiles": str, 
    }
    
    # Process data in batches
    all_predictions = []
    smiles_list = []
    
    for i in range(0, len(test_dataset), args.batch_size):
        batch = test_dataset[i:i + args.batch_size]
        # Process SMILES
        smiles = batch[args.data_column_name]
        smiles_list.extend(smiles)
        
        # Tokenize
        tokenized = tokenizer(
            smiles,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        # Process graph data if present
        graph_batch = None
        if 'ba_edge_index' in batch.keys():
            # Create a list of dictionaries for each sample in the batch
            batch_data = []
            for i in range(len(batch['smiles'])):  # Use length of any feature to get batch size
                sample_data = {
                    key: convert_value(key, value[i], type_map) if isinstance(value, list) else value
                    for key, value in batch.items()
                    if key in type_map  # Only include graph-related features
                }
                batch_data.append(Data(**sample_data))
            graph_batch = TriBatch.from_data_list(batch_data)
        else:
            logger.warning("No graph data found in batch")
        
        # Move tensors to the same device as model
        device = next(model.parameters()).device
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        if graph_batch is not None:
            graph_batch = graph_batch.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=graph_batch
            )

            predictions = outputs['logits'].cpu().numpy()
            all_predictions.extend(predictions)
    
    # Load scaler if provided
    if args.scaler_path:
        scaler = joblib.load(args.scaler_path)
        predictions = scaler.inverse_transform(all_predictions).flatten()
    else:
        predictions = all_predictions
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    df = pd.DataFrame({
        "smiles": smiles_list,
        "prediction": predictions
    })
    output_path = os.path.join(args.output_dir, "test_predictions.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved predictions to {output_path}")

if __name__ == "__main__":
    main()
