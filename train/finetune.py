import logging
import os
import sys
import random
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
        metadata={"help": ("Specify the model's output size.")},
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
    allow_smiles_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Allow the model to only use SMILES as input."},
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
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "jsonl",
                    "txt",
                ], "`train_file` should be a csv, a jsonl or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "jsonl",
                    "txt",
                ], "`validation_file` should be a csv, a jsonl or a txt file."
            if self.test_files is not None:
                extension = self.test_files[0].split(".")[-1]
                assert extension in [
                    "csv",
                    "jsonl",
                    "txt",
                ], "`test_file` should be a csv, a jsonl or a txt file."


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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    ## Start load data.
    if True:
        data_files = {}
        dataset_args = {}

        # Check if loading from Hugging Face Hub or local files
        if data_args.dataset_name is not None:
            # Load from Hugging Face Hub
            print(f"Loading dataset from Hugging Face Hub: {data_args.dataset_name}")
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                streaming=data_args.streaming,
                cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
                use_auth_token=True if model_args.use_auth_token else None,
            )
            # For Hub datasets, we assume they are already in the correct format (jsonl/json)
            extension = "jsonl"
        else:
            # Load from local files (existing behavior)
            if data_args.train_files is not None:
                data_files["train"] = data_args.train_files
            if data_args.validation_files is not None:
                data_files["validation"] = data_args.validation_files
            if data_args.test_files is not None:
                data_files["test"] = data_args.test_files

            # Identify extension
            if "train" in data_files and len(data_files["train"]) > 0:
                extension = data_files["train"][0].split(".")[-1]
            elif "validation" in data_files and len(data_files["validation"]) > 0:
                extension = data_files["validation"][0].split(".")[-1]
            else:
                raise ValueError(
                    "No valid training or validation files found to determine the extension."
                )

            # --- Debugging --hard code here ------------------------------------------------------------
            # if extension == "csv":
            #     extension = "jsonl"
            # --- Debugging --hard code here ------------------------------------------------------------

        # Only process local files (Hub datasets are already processed)
        if data_args.dataset_name is None and extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
            # For txt files, we need to process SMILES into graph data
            from preprocess.mol3d_processor import smiles2GeoGraph

            def process_smiles_to_graph(example):
                smiles = example["smiles"]
                graph_data = smiles2GeoGraph(smiles, brics=False, geo_operation=False)
                if graph_data is None and model_args.allow_smiles_only:
                    return {"smiles": smiles}
                elif graph_data is None:
                    return None

                return {
                    "smiles": smiles,
                    "x": graph_data.x.tolist(),
                    "edge_index": graph_data.edge_index.tolist(),
                    "edge_attr": graph_data.edge_attr.tolist(),
                }

            # Load the text dataset first
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                streaming=data_args.streaming,
                cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
                **dataset_args,
            )

            # Rename 'text' column to 'smiles'
            raw_datasets = raw_datasets.rename_column("text", "smiles")

            # Process SMILES to graph data
            raw_datasets = raw_datasets.map(
                process_smiles_to_graph,
                remove_columns=raw_datasets["train"].column_names,
                desc="Converting SMILES to graph data",
            )
            # Filter out None values (invalid SMILES)
            raw_datasets = raw_datasets.filter(lambda x: x is not None)
        elif data_args.dataset_name is None and extension == "csv":
            # For CSV files, we need to process SMILES into graph data
            from preprocess.mol3d_processor import smiles2GeoGraph

            def process_smiles_to_graph(example):
                smiles = example["smiles"]
                graph_data = smiles2GeoGraph(smiles, brics=False, geo_operation=False)
                if graph_data is None and model_args.allow_smiles_only:
                    return {"smiles": smiles}
                elif graph_data is None:
                    return None
                # Create a new dict with graph data while preserving all other columns
                result = {
                    "smiles": smiles,
                    "x": graph_data.x.tolist(),
                    "edge_index": graph_data.edge_index.tolist(),
                    "edge_attr": graph_data.edge_attr.tolist(),
                }
                # Preserve all other columns (including labels)
                for key, value in example.items():
                    if key not in result:
                        result[key] = value
                return result

            # Load the CSV dataset
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                streaming=data_args.streaming,
                cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
                **dataset_args,
            )

            # Process SMILES to graph data
            raw_datasets = raw_datasets.map(
                process_smiles_to_graph, desc="Converting SMILES to graph data"
            )
            # Filter out None values (invalid SMILES)
            raw_datasets = raw_datasets.filter(lambda x: x is not None)
        elif data_args.dataset_name is None:
            # Load from local JSON/JSONL files
            if extension == "jsonl":
                extension = "json"
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                streaming=data_args.streaming,
                cache_dir=os.path.join(training_args.output_dir, "dataset_cache"),
                **dataset_args,
            )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        # Only apply this for local file loading, not Hub datasets
        if data_args.dataset_name is None and "validation" not in raw_datasets.keys():
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

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    multi_label_mark = False
    if len(column_names) == 1:
        input_column_name = "smiles" if "smiles" in column_names else column_names[0]
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

    # --------------- Normalization Labels new ----------------
    if model_args.task_type == "regression" and data_args.normlization:
        # -- 1. Prepare scaler (using training set only)
        if not multi_label_mark:  # Single task
            y_train = np.array(raw_datasets["train"][target_column_name]).reshape(-1, 1)
        else:  # Multi task
            target_cols = target_column_name
            y_train = np.column_stack([raw_datasets["train"][c] for c in target_cols])

        scaler = StandardScaler().fit(y_train)

        # -- 2. Define normalization function
        if not multi_label_mark:

            def normalize(batch):
                y = np.array(batch[target_column_name]).reshape(-1, 1)
                batch[target_column_name] = scaler.transform(y).flatten().tolist()
                return batch

        else:

            def normalize(batch):
                y = np.column_stack([batch[c] for c in target_cols])
                y_scaled = scaler.transform(y)
                for i, c in enumerate(target_cols):
                    batch[c] = y_scaled[:, i].tolist()
                return batch

        # -- 3. Apply
        raw_datasets = raw_datasets.map(
            normalize, batched=True, load_from_cache_file=False
        )

        # -- 4. Oversample (as needed)
        if data_args.over_sample:
            from preprocess.oversample import sample_operation

            raw_datasets = sample_operation(raw_datasets, -1, 1)
    # --------------- Normalization Labels new ----------------

    # Weight labels
    if (
        model_args.task_type == "classification"
        and data_args.class_weight
        and multi_label_mark == False
    ):
        labels = raw_datasets["train"][target_column_name]
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(labels), y=labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        model.class_weight = class_weights

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from scipy.stats import pearsonr
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            precision_recall_fscore_support,
            accuracy_score,
            precision_recall_curve,
            auc,
        )

        if model_args.task_type == "regression":
            if multi_label_mark == False:

                def compute_metrics(pred):
                    labels = pred.label_ids
                    predictions = pred.predictions

                    if data_args.normlization:
                        predictions = scaler.inverse_transform(
                            predictions.reshape(-1, 1)
                        )
                        labels = scaler.inverse_transform(labels.reshape(-1, 1))

                    predictions = predictions.flatten().tolist()
                    labels = labels.flatten().tolist()

                    if any([x is None or np.isnan(x) for x in predictions + labels]):
                        return {
                            "mae": float("nan"),
                            "rmse": float("nan"),
                            "r2": float("nan"),
                            "pearson_corr": float("nan"),
                        }

                    mae = mean_absolute_error(labels, predictions)
                    rmse = np.sqrt(mean_squared_error(labels, predictions))
                    r2 = r2_score(labels, predictions)

                    try:
                        pearson_corr, _ = pearsonr(labels, predictions)
                    except ValueError:
                        pearson_corr = float("nan")

                    return {
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                        "pearson_corr": pearson_corr,
                    }

            elif multi_label_mark == True:
                # === Multi-task / Multi-target regression version ===
                def compute_metrics(pred):
                    labels = pred.label_ids  # shape = [N, T]
                    predictions = pred.predictions  # shape = [N, T]

                    # Force 2-D
                    labels = np.asarray(labels)
                    predictions = np.asarray(predictions)

                    # Inverse normalization (restore by column)
                    if data_args.normlization:
                        predictions = scaler.inverse_transform(predictions)
                        labels = scaler.inverse_transform(labels)

                    # Per-task metrics
                    metrics = {}
                    mae_list, rmse_list, r2_list, p_list = [], [], [], []
                    for t in range(predictions.shape[1]):
                        y_true = labels[:, t]
                        y_pred = predictions[:, t]

                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2 = r2_score(y_true, y_pred)
                        try:
                            p, _ = pearsonr(y_true, y_pred)
                        except ValueError:
                            p = float("nan")

                        # Save to list for macro average
                        mae_list.append(mae)
                        rmse_list.append(rmse)
                        r2_list.append(r2)
                        p_list.append(p)

                        # Output with task name prefix
                        task_name = (
                            target_cols[t] if t < len(target_cols) else f"task{t}"
                        )
                        metrics.update(
                            {
                                f"{task_name}_mae": mae,
                                f"{task_name}_rmse": rmse,
                                f"{task_name}_r2": r2,
                                f"{task_name}_pearson": p,
                            }
                        )

                    # Macro average
                    metrics.update(
                        {
                            "mae": np.mean(mae_list),
                            "rmse": np.mean(rmse_list),
                            "r2": np.mean(r2_list),
                            "pearson_corr": np.nanmean(p_list),  # Filter nan values
                        }
                    )
                    return metrics

        elif model_args.task_type == "classification":
            if model_args.output_size == -1:

                def compute_metrics(pred, threshold=0.5, fig=False):
                    labels = pred.label_ids

                    probs = np.exp(pred.predictions) / np.sum(
                        np.exp(pred.predictions), axis=-1, keepdims=True
                    )
                    if probs.shape[1] == 2:
                        preds = (probs[:, 1] >= threshold).astype(int)
                    else:
                        preds = np.argmax(pred.predictions, axis=-1)

                    # Accuracy, Precision, Recall, F1
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels, preds, average="weighted"
                    )
                    accuracy = accuracy_score(labels, preds)
                    try:
                        if len(np.unique(labels)) == 2:
                            roc_auc = roc_auc_score(labels, probs[:, 1])
                        else:
                            roc_auc = roc_auc_score(
                                labels, probs, multi_class="ovo", average="weighted"
                            )
                    except ValueError:
                        roc_auc = float("nan")

                    try:
                        if len(np.unique(labels)) == 2:
                            pr_auc = average_precision_score(labels, probs[:, 1])
                        else:
                            pr_auc = average_precision_score(
                                labels, probs, average="weighted"
                            )
                    except ValueError:
                        pr_auc = float("nan")

                    return {
                        "accuracy": accuracy,
                        "f1": f1,
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "precision": precision,
                        "recall": recall,
                    }

            elif model_args.output_size >= 2:

                def compute_metrics(pred, threshold=0.5, fig=False):
                    labels = (
                        pred.label_ids
                    )  # Ground truth labels, shape: [batch_size, num_classes]
                    probs = (
                        pred.predictions
                    )  # Model probabilities/logits, shape: [batch_size, num_classes]

                    # Apply sigmoid to logits to get probabilities (required for multi-label classification)
                    probs = 1 / (1 + np.exp(-probs))

                    # Predict labels based on the threshold
                    preds = (probs >= threshold).astype(int)

                    # Calculate metrics for multi-label classification
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels, preds, average="samples", zero_division=0
                    )  # "samples" average for multi-label tasks

                    # Micro-average accuracy (not per-sample accuracy)
                    task_accuracies = np.mean(
                        labels == preds, axis=0
                    )  # Accuracy for each task
                    accuracy = np.mean(
                        task_accuracies
                    )  # Average accuracy across all tasks

                    # ROC-AUC (multi-label)
                    try:
                        roc_auc = roc_auc_score(
                            labels, probs, average="macro", multi_class="ovr"
                        )
                    except ValueError:
                        roc_auc = float("nan")

                    # PR-AUC (multi-label)
                    try:
                        pr_auc = average_precision_score(labels, probs, average="macro")
                    except ValueError:
                        pr_auc = float("nan")

                    return {
                        "accuracy": accuracy,
                        "f1": f1,
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "precision": precision,
                        "recall": recall,
                    }

            else:
                print("Model args output_size is not correct.")

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
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Handle labels
        if isinstance(target_column_name, str):
            labels = torch.stack(
                [torch.tensor(item[target_column_name]) for item in batch]
            )
        elif isinstance(target_column_name, list):
            labels = torch.stack(
                [
                    torch.tensor(
                        [
                            (
                                0
                                if (col_value := item[col]) != col_value
                                or col_value is None
                                else col_value
                            )
                            for col in target_column_name
                        ]
                    )
                    for item in batch
                ]
            )

        # Process graph data
        def convert_value(key, value):
            if key in type_map:
                converter = type_map[key]
                if callable(converter):
                    return converter(value)
                else:
                    return torch.tensor(value, dtype=converter)
            return value

        batch_data = [
            Data(
                **{
                    key: convert_value(key, value) if isinstance(value, list) else value
                    for key, value in data_dict.items()
                }
            )
            for data_dict in batch
        ]
        if batch_data[0].x is not None:
            graph_batch = TriBatch.from_data_list(batch_data)
        else:
            graph_batch = None

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": labels,
            "graph_data": graph_batch,
        }

    # Initialize our Trainer
    training_args.remove_unused_columns = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
        compute_metrics=(
            compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        print(training_args.local_rank, "start train")

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

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

        metrics = trainer.evaluate(ignore_keys=["pooled_output"])

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Test
        logger.info("*** Test ***")
        test_results = trainer.predict(test_dataset)
        print(test_results.metrics)
        test_metrics = test_results.metrics
        test_metrics["test_samples"] = len(test_dataset)

        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
