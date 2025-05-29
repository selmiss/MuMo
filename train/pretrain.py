import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import datasets
import torch
from datasets import load_dataset
from datasets import Dataset
from datasets.combine import interleave_datasets
from torch_geometric.data import Data
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
from graph_batch import TriBatch
import tempfile

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_class: Optional[str] = field(
        default=None,
        metadata={"help": ("The model class you want to load")},
    )
    task_type: Optional[str] = field(
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
    block_size: Optional[int] = field(
        default=512,
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
        default=1,
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
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                    "pt",
                    "jsonl"
                ], "`train_file` should be a csv, a json, a pt, or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                    "pt",
                    "jsonl"
                ], "`validation_file` should be a csv, a json, a pt, or a txt file."

from transformers import DataCollatorForLanguageModeling
from torch_geometric.data import Batch
import torch


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

class CustomDataCollatorForLanguageAndGraph(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length=512, mlm=True, mlm_probability=0.15):
        """
        A custom collator that combines the functionality of DataCollatorForLanguageModeling
        and additional processing for custom data (e.g., graph structures).
        
        Args:
            tokenizer: Pretrained tokenizer for text data.
            max_length: Maximum sequence length for tokenized text.
            mlm: Whether to use Masked Language Modeling.
            mlm_probability: Masking probability for MLM.
        """
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.max_length = max_length

    def __call__(self, batch):
        """
        Process a batch of samples, combining text, graph, and label data.
        """
        # Normal smiles processor
        smiles = [item["smiles"] for item in batch]
        tokenized = self.tokenizer(
            smiles,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",  # Output PyTorch tensors
        )
        input_ids = torch.tensor(tokenized["input_ids"])
        attention_mask = torch.tensor(tokenized["attention_mask"])
        
        # Apply MLM (if enabled) using the parent class's method
        if self.mlm:
            input_ids, labels = self.torch_mask_tokens(input_ids)
        else:
            labels = input_ids.clone()
        for i, label in enumerate(labels):
            if (label != -100).sum() == 0:
                seq_len = input_ids[i].size(0)
                random_idx = torch.randint(0, seq_len, (1,)).item()
                labels[i][random_idx] = input_ids[i][random_idx]
                input_ids[i][random_idx] = self.tokenizer.mask_token_id
        
        
        # Graph data processor
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

        # Return combined output
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "graph_data": graph_batch,
        }


def main():

    ### Start Load Config
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
    ### End Load Config

    ### Start Load Logging
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

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    ### End Load logging

    ### Start Load tokenizer and config
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
        raise ValueError("You need to specify the model config.")

    print(training_args.local_rank, "start load tokenizer")
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
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
        )

    print(training_args.local_rank, "end load tokenizer")
    ### End Load tokenizer and config

    ### Start Load Data Files
    set_seed(training_args.seed)
    data_files = {}
    dataset_args = {}

    # Train datasets
    if data_args.train_files is not None:
        if isinstance(data_args.train_files, list) and len(data_args.train_files) > 0:
            print(data_args.train_files)
            data_files["train"] = data_args.train_files
            print("Number of train files:", len(data_args.train_files))
        else:
            print("Warning: train_files is empty or not a list.")
    else:
        print("Warning: train_files is None.")

    # Evaluation datasets
    if data_args.validation_files is not None:
        if (
            isinstance(data_args.validation_files, list)
            and len(data_args.validation_files) > 0
        ):
            data_files["validation"] = data_args.validation_files
        else:
            print("Warning: validation_files is empty or not a list.")
    else:
        print("Warning: validation_files is None.")

    # Identify extentions
    if "train" in data_files and len(data_files["train"]) > 0:
        extension = data_files["train"][0].split(".")[-1]
    elif "validation" in data_files and len(data_files["validation"]) > 0:
        extension = data_files["validation"][0].split(".")[-1]
    else:
        raise ValueError(
            "No valid training or validation files found to determine the extension."
        )
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            streaming=data_args.streaming,
            cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
            **dataset_args,
        )
    elif extension == "csv":
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            streaming=data_args.streaming,
            cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
            **dataset_args,
        )
    else:
        if extension == "jsonl": 
            extension = "json"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            streaming=data_args.streaming,
            cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
            **dataset_args,
        )

    if data_args.streaming:
        raw_datasets = raw_datasets.shuffle(
            seed=training_args.seed, buffer_size=1000000
        )

    # Process SMILES to graph data
    from preprocess.mol3d_processor import smiles2GeoGraph
    import torch

    def process_smiles_to_graph(example):
        # For JSONL files, we don't need to process the SMILES
        if extension == "jsonl":
            return example
            
        smiles = example['smiles']
        graph_data = smiles2GeoGraph(smiles, brics=False, geo_operation=False)
        if graph_data is None:
            return None
        # Create a new dict with graph data while preserving all other columns
        result = {
            'smiles': smiles,
            'x': graph_data.x.tolist(),
            'edge_index': graph_data.edge_index.tolist(),
            'edge_attr': graph_data.edge_attr.tolist(),
            'cluster_idx': torch.zeros(graph_data.x.shape[0], dtype=torch.int64).tolist()
        }
        # Preserve all other columns
        for key, value in example.items():
            if key not in result:
                result[key] = value
        return result

    # Process the datasets
    if training_args.do_train:
        raw_datasets["train"] = raw_datasets["train"].map(
            process_smiles_to_graph,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Processing train dataset"
        )
    
    if training_args.do_eval:
        raw_datasets["validation"] = raw_datasets["validation"].map(
            process_smiles_to_graph,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Processing validation dataset"
        )

    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
    ### End Load Data Files

    ### Start Load Model
    from model.load_model import load_model, initialize_weights

    ### End Load Model
    model = load_model(config, tokenizer, training_args, model_args)

    print(training_args.local_rank, "start select train_dataset")
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None and data_args.streaming == False:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    print(training_args.local_rank, "end select train_dataset")

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        print(training_args.local_rank, "start select eval_dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None and data_args.streaming == False:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        print(training_args.local_rank, "end select eval_dataset")
    ### End Process Data


    print(training_args.local_rank, "Initialize our Trainer")
    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=CustomDataCollatorForLanguageAndGraph(
            tokenizer=tokenizer,
            max_length=data_args.block_size,
            mlm=True,
            mlm_probability=0.15,
        ),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        print(training_args.local_rank, "start train")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
