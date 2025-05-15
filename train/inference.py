import logging
import os
import sys
import random
import pandas as pd
import joblib
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import datasets
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
import transformers
from sklearn.utils.class_weight import compute_class_weight
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch_geometric.data import Data


from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
from sklearn.metrics import accuracy_score


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = (
        field(
            default=None,
            metadata={
                "help": (
                    "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
                )
            },
        ),
    )
    model_class: Optional[str] = field(
        default=None,
        metadata={"help": ("The model class you want to load")},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    task_type: Optional[str] = field(
        default="regression",
        metadata={"help": "regression or classification"},
    )
    frozen_layer: Optional[int] = field(
        default=-2,
        metadata={
            "help": (
                "Freeze the model's layer. None for no, -1 for all, 0 for only embedding, others define the backbone layers."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    output_size: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "Specify the model's output size."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    pool_method: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_files: Optional[List[str]] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_files: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    normlization: Optional[bool] = field(
        default=False, metadata={"help": "Using normlization for data's label"}
    )
    over_sample: Optional[bool] = field(
        default=False, metadata={"help": "Sampling the tail data"}
    )
    test_files: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to test the perplexity on (a text file)."
        },
    )
    data_column_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Column name of the input data of the model"},
    )
    label_column_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Column name of the input data's labels of the model"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    class_weight: bool = field(
        default=False, metadata={"help": "Use class_weight for classification dataset"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=10,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )

        if (
            self.dataset_name is None
            and self.train_files is None
            and self.validation_files is None
        ):
            pass
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."
            if self.test_files is not None:
                extension = self.test_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`test_file` should be a csv, a json or a txt file."


type_map = {
    "x": torch.float32,
    "edge_index": torch.int64,
    "edge_attr": torch.float32,
    'ba_edge_index': torch.int64, 
    'ba_edge_attr': torch.float32, 
    'fra_edge_index':  torch.int64, 
    'fra_edge_attr': torch.float32, 
    'cluster_idx': torch.int64, 
    'bafra_edge_index': torch.int64, 
    'bafra_edge_attr':  torch.float32,
    "smiles": str, 
}

def main():


    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_clm", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    ## Start load data.
    if True:
        data_files = {}
        dataset_args = {}
        if data_args.test_files is not None:
            data_files["test"] = data_args.test_files
        extension = "csv"
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        if extension == "csv":
            extension = "json"
            
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
            **dataset_args,
        )
    

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "padding_side": "right",
    }
    
    ## Load tokenizer.
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
        
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ### Start Load Model
    if model_args.pool_method is not None:
        config.pool_method = model_args.pool_method
    if model_args.model_class is None and model_args.model_name_or_path:
        raise ValueError("Please use model_class to load model.")

    elif model_args.model_class is not None:
        from model.load_model import load_model

        config.output_size = model_args.output_size
        model = load_model(config, tokenizer, training_args, model_args)

        model.initialize_parameters()
    else:
        raise ValueError(
            "For finetuning stage, you have to load a exiting pretrained model."
        )

    column_names = list(raw_datasets["test"].features)

    multi_label_mark = False
    if len(column_names) == 1:
        text_column_name = "text" if "text" in column_names else column_names[0]
    elif len(column_names) >= 2:
        if data_args.data_column_name is not None:
            input_column_name = data_args.data_column_name
        else:
            input_column_name = "input" if "input" in column_names else column_names[0]
        if data_args.label_column_name is not None:
            if len(data_args.label_column_name) == 1:
                target_column_name = data_args.label_column_name[0]
            else:
                multi_label_mark = True
                target_column_name = data_args.label_column_name
        else:
            target_column_name = (
                "target" if "target" in column_names else column_names[1]
            )
    else:
        raise ValueError("Column of data file is not correct.")

    print("train_on_inputs", data_args.train_on_inputs)

    test_dataset = raw_datasets["test"]
    
    from torch_geometric.data import Batch
    from graph_batch import TriBatch

    def custom_collate_fn(batch):
        """
        Custom collate function to merge text tensors and graph data into batches.
        Args:
            batch (list): List of samples, each containing tokenized text and graph data.

        Returns:
            A dictionary with batched text tensors, graph data, and labels.
        """
        # Collect text data into batched tensors
        
        smiles = [item["smiles"] for item in batch]
        tokenized = tokenizer(
                    smiles,
                    truncation=True,
                    max_length=data_args.block_size,
                    padding=True,
                    return_tensors=None,
                )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
            
        # Collect graph data and batch them using PyTorch Geometric's Batch
        if 'ba_edge_index' in batch[0].keys():
            def convert_value(key, value):
                if key in type_map:
                    converter = type_map[key]
                    if callable(converter):
                        return converter(value)
                    else:
                        return torch.tensor(value, dtype=converter)
                return value 
            
            batch_data = [
                Data(**{key: convert_value(key, value) if isinstance(value, list) else value for key, value in data_dict.items()})
                for data_dict in batch
            ]
            
            graph_batch = TriBatch.from_data_list(batch_data)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "graph_data": graph_batch  # Batched graph data
        }
    
    # Initialize our Trainer
    training_args.remove_unused_columns = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
        compute_metrics=None
    )

    # Test
    logger.info("*** Test ***")

    # Run prediction
    test_results = trainer.predict(test_dataset)

    # Load scaler
    scaler_path = "xxx/Nips/dataset/dk/scaler.pkl"
    scaler = joblib.load(scaler_path)

    # Inverse normalization
    raw_preds = test_results.predictions
    if raw_preds.ndim == 2 and raw_preds.shape[1] == 1:
        raw_preds = raw_preds.reshape(-1, 1)
    predictions = scaler.inverse_transform(raw_preds).flatten()

    # Extract smiles (assumes raw_datasets["test"] has the same order)
    smiles_list = raw_datasets["test"]["smiles"]

    # Save to CSV
    df = pd.DataFrame({
        "smiles": smiles_list,
        "prediction": predictions
    })
    output_path = os.path.join(training_args.output_dir, "test_predictions.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved predictions to {output_path}")




if __name__ == "__main__":
    main()
